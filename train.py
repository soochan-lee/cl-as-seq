import os
import os.path as path
import shutil
import socket
from argparse import ArgumentParser
from datetime import datetime
from glob import glob
from modulefinder import ModuleFinder

import math
import torch
import yaml
from einops import rearrange, pack
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset import DATASET
from models import MODEL
from utils import Timer

parser = ArgumentParser()
parser.add_argument('--model-config', '-mc', required=True)
parser.add_argument('--data-config', '-dc', required=True)
parser.add_argument('--log-dir', '-l')
parser.add_argument('--override', '-o', default='')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--no-backup', action='store_true')

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_config(config_path):
    with open(config_path, 'r') as f:
        new_config = yaml.full_load(f)
    config = {}
    if 'include' in new_config:
        include_config = get_config(new_config['include'])
        config.update(include_config)
        del new_config['include']
    config.update(new_config)
    return config


def main():
    if torch.cuda.is_available():
        print(f'Running on {socket.gethostname()} | {torch.cuda.device_count()}x {torch.cuda.get_device_name()}')
    args = parser.parse_args()

    # Load config
    config = get_config(args.model_config)
    data_config = get_config(args.data_config)
    config.update(data_config)

    # Override options
    for option in args.override.split('|'):
        if not option:
            continue
        address, value = option.split('=')
        keys = address.split('.')
        here = config
        for key in keys[:-1]:
            if key not in here:
                here[key] = {}
            here = here[key]
        if keys[-1] not in here:
            print(f'Warning: {address} is not defined in config file.')
        here[keys[-1]] = yaml.load(value, Loader=yaml.FullLoader)

    if 'y_vocab' in config and config['y_vocab'] is None:
        config['y_vocab'] = config['tasks']

    # Prevent overwriting
    config['log_dir'] = args.log_dir
    config_save_path = path.join(config['log_dir'], 'config.yaml')
    try:
        # Try to open config file to bypass NFS cache
        with open(config_save_path, 'r') as f:
            f.read(1)
            config_exists = True
    except FileNotFoundError:
        config_exists = False

    if config_exists and not args.resume:
        print(f'WARNING: {args.log_dir} already exists. Skipping...')
        exit(0)

    # Save config
    os.makedirs(config['log_dir'], mode=0o755, exist_ok=True)
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    print(f'Config saved to {config_save_path}')

    # Save code
    if not args.no_backup:
        code_dir = path.join(config['log_dir'], 'code_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
        mf = ModuleFinder([os.getcwd()])
        mf.run_script(__file__)
        for name, module in mf.modules.items():
            if module.__file__ is None:
                continue
            rel_path = path.relpath(module.__file__)
            new_path = path.join(code_dir, rel_path)
            new_dirname = path.dirname(new_path)
            os.makedirs(new_dirname, mode=0o750, exist_ok=True)
            shutil.copy2(rel_path, new_path)
        print(f'Code saved to {code_dir}')

    # Get a free port for DDP
    sock = socket.socket()
    sock.bind(('', 0))
    ddp_port = sock.getsockname()[1]
    sock.close()

    # Start DDP
    world_size = torch.cuda.device_count()
    assert config['batch_size'] % world_size == 0, 'Batch size must be divisible by the number of GPUs.'
    config['batch_size'] //= world_size
    assert config['eval_batch_size'] % world_size == 0, 'Eval batch size must be divisible by the number of GPUs.'
    config['eval_batch_size'] //= world_size
    mp.spawn(train, args=(world_size, ddp_port, args, config), nprocs=world_size)


def train(rank, world_size, port, args, config):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # Initialize process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    writer = None
    if rank == 0:
        writer = SummaryWriter(config['log_dir'], flush_secs=15)

    # Build model
    model = MODEL[config['model']](config).to(rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])
    optim = getattr(torch.optim, config['optim'])(model.parameters(), **config['optim_args'])
    lr_sched = getattr(lr_scheduler, config['lr_sched'])(optim, **config['lr_sched_args'])
    start_step = 0

    # Resume checkpoint
    if args.resume:
        ckpt_paths = sorted(glob(path.join(config['log_dir'], 'ckpt-*.pt')))
        if len(ckpt_paths) > 0:
            ckpt_path = ckpt_paths[-1]
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt['model'])
            optim.load_state_dict(ckpt['optim'])
            lr_sched.load_state_dict(ckpt['lr_sched'])
            # Get step number from checkpoint name
            start_step = int(path.basename(ckpt_path).split('-')[1].split('.')[0])
            print(f'Checkpoint loaded from {ckpt_path}')
    optim.zero_grad()

    # Data
    Dataset = DATASET[config['dataset']]
    train_set = Dataset(config, root='./data', meta_split='train')
    test_set = Dataset(config, root='./data', meta_split='test')
    train_loader = DataLoader(
        train_set,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'])
    test_loader = DataLoader(
        test_set,
        batch_size=config['eval_batch_size'],
        num_workers=config['num_workers'])
    train_loader_iter = iter(train_loader)
    test_loader_iter = iter(test_loader)

    # Main training loop
    start_time = datetime.now()
    print(f'Training started at {start_time}')
    eval_num_bites = 1
    for step in range(start_step + 1, config['max_train_steps'] + 1):
        train_x, train_y, test_x, test_y = next(train_loader_iter)

        batch_size = train_x.shape[0]
        digested = 0
        outputs = []
        while batch_size - digested > 0:
            # Gradient accumulation
            bite = min(batch_size - digested, math.ceil(config['batch_size'] / config['num_bites']))
            train_x_bite = train_x[digested:digested + bite].to(rank)
            train_y_bite = train_y[digested:digested + bite].to(rank)
            test_x_bite = test_x[digested:digested + bite].to(rank)
            test_y_bite = test_y[digested:digested + bite].to(rank)
            if batch_size - digested - bite == 0:
                # Last bite
                output = forward_backward(
                    model, train_x_bite, train_y_bite, test_x_bite, test_y_bite,
                    batch_size=batch_size, config=config, evaluate=step % config['summary_interval'] == 0)
            else:
                with model.no_sync():
                    output = forward_backward(
                        model, train_x_bite, train_y_bite, test_x_bite, test_y_bite,
                        batch_size=batch_size, config=config, evaluate=False)

            outputs.append(output)
            digested += bite

        optim.step()
        lr_sched.step()
        optim.zero_grad()

        if 'attn_loss' in config and config['attn_loss'] > 0 and step >= config['attn_loss_steps']:
            config['attn_loss'] = 0
            print('\nTurning off attention loss')
        if 'distributed_loss' in config and config['distributed_loss'] and step >= config['distributed_loss_steps']:
            config['distributed_loss'] = False
            print('\nTurning off distributed loss')

        if rank == 0 and config['input_type'] == 'image' and step == config['summary_interval'] \
                and config['tasks'] * config['train_shots'] <= 100:
            # Summarize meta-train images once
            train_x_summ = rearrange(train_x[0], '(tasks s) c h w -> (s tasks) c h w', tasks=config['tasks'])
            test_x_summ = rearrange(test_x[0], '(tasks s) c h w -> (s tasks) c h w', tasks=config['tasks'])
            writer.add_image(
                'meta-train/train',
                make_grid(train_x_summ, nrow=config['tasks']), step)
            writer.add_image(
                'meta-train/test',
                make_grid(test_x_summ, nrow=config['tasks']), step)

        if step % config['summary_interval'] == 0:
            loss_mean = sum([output['loss_mean'] * output['proportion'] for output in outputs])
            gathered_loss_mean = torch.zeros(world_size, dtype=loss_mean.dtype, device=loss_mean.device)
            dist.all_gather_into_tensor(gathered_loss_mean, loss_mean.detach())
            loss_mean = gathered_loss_mean.mean()
            if rank == 0:
                writer.add_scalar('loss/train', loss_mean.item(), step)
                writer.add_scalar('lr', lr_sched.get_last_lr()[0], step)

            if 'attn_losses' in outputs[0]:
                # Combine attention losses from different bites
                layer_attn_loss = [attn_loss * outputs[0]['proportion'] for attn_loss in outputs[0]['attn_losses']]
                for output in outputs[1:]:
                    for layer, attn_loss in enumerate(output['attn_losses']):
                        layer_attn_loss[layer] += attn_loss * output['proportion']

                # Summarize attention loss of each layer
                for layer, attn_loss in enumerate(layer_attn_loss):
                    gathered_attn_loss = torch.zeros(world_size, dtype=attn_loss.dtype, device=attn_loss.device)
                    dist.all_gather_into_tensor(gathered_attn_loss, attn_loss)
                    if rank == 0:
                        writer.add_scalar(f'loss_attn/layer{layer}', gathered_attn_loss.mean().item(), step)

            if rank == 0 and 'inner_lr' in outputs[0]:
                writer.add_scalar('lr_inner', outputs[0]['inner_lr'].item(), step)

            # Compute train accuracy
            if 'evaluation' in outputs[0]:
                evaluation = torch.cat([output['evaluation'] for output in outputs], dim=0)
                acc_train = evaluation.float().mean()
                gathered_acc_train = torch.zeros(world_size, dtype=acc_train.dtype, device=acc_train.device)
                dist.all_gather_into_tensor(gathered_acc_train, acc_train)
                if rank == 0:
                    writer.add_scalar('acc/train', gathered_acc_train.mean().item(), step)

            # Compute remaining time
            if rank == 0:
                now = datetime.now()
                elapsed_time = now - start_time
                elapsed_steps = step - start_step
                total_steps = config['max_train_steps'] - start_step
                est_total = elapsed_time * total_steps / elapsed_steps
                # Remove microseconds for brevity
                elapsed_time = str(elapsed_time).split('.')[0]
                est_total = str(est_total).split('.')[0]
                print(f'\r[Step {step}] [{elapsed_time} / {est_total}] Loss: {loss_mean:.8f}', end='')

            if torch.isnan(loss_mean).any().item():
                raise RuntimeError('NaN loss encountered')

        if rank == 0 and step % config['ckpt_interval'] == 0:
            # Remove old checkpoints
            ckpt_paths = sorted(glob(path.join(config['log_dir'], 'ckpt-*.pt')))
            for ckpt_path in ckpt_paths[:-1]:
                os.remove(ckpt_path)

            new_ckpt_path = path.join(config['log_dir'], f'ckpt-{step:06}.pt')
            torch.save({
                'step': step,
                'config': config,
                'model': model.state_dict(),
                'optim': optim.state_dict(),
                'lr_sched': lr_sched.state_dict(),
            }, new_ckpt_path)
            print(f'\nCheckpoint saved to {new_ckpt_path}')

        if step % config['eval_interval'] == 0:
            # Meta-test
            print()
            model.eval()
            with torch.no_grad(), Timer('Evaluation time: {:.3f}s'):
                loss_mean = 0
                correct, total = 0, 0
                eval_size = config['eval_iters'] * config['eval_batch_size']
                for _ in range(config['eval_iters']):
                    train_x, train_y, test_x, test_y = next(test_loader_iter)
                    train_x, train_y = train_x.to(model.device), train_y.to(model.device)
                    test_x, test_y = test_x.to(model.device), test_y.to(model.device)

                    batch_size = train_x.shape[0]
                    digested = 0
                    while batch_size - digested > 0:
                        bite = min(batch_size - digested, math.ceil(config['eval_batch_size'] / config['num_bites']))
                        train_x_bite = train_x[digested:digested + bite].to(rank)
                        train_y_bite = train_y[digested:digested + bite].to(rank)
                        test_x_bite = test_x[digested:digested + bite].to(rank)
                        test_y_bite = test_y[digested:digested + bite].to(rank)

                        try:
                            output = forward(
                                model, train_x_bite, train_y_bite, test_x_bite, test_y_bite,
                                eval_size=eval_size)
                        except RuntimeError as e:
                            if 'CUDA out of memory' in str(e):
                                eval_num_bites += 1
                                if eval_num_bites > batch_size:
                                    raise RuntimeError('Even a bite size of 1 is too large')
                                print(f'\n{bite}/{batch_size} is too large for an evaluation bite. '
                                      f'Increasing the number of bites to {eval_num_bites}')
                                torch.cuda.empty_cache()
                                continue
                            else:
                                raise e
                        digested += bite

                        loss_mean += output['loss_mean'] * output['proportion']
                        if 'correct' in output:
                            correct += output['correct']
                            total += output['total']

                gathered_loss_mean = torch.zeros(world_size, dtype=loss_mean.dtype, device=loss_mean.device)
                dist.all_gather_into_tensor(gathered_loss_mean, loss_mean)
                loss_mean = gathered_loss_mean.mean().item()
                if rank == 0:
                    writer.add_scalar('loss/test', loss_mean, step)

                if total > 0:
                    gathered_correct = torch.zeros(world_size, dtype=correct.dtype, device=correct.device)
                    gathered_total = torch.zeros(world_size, dtype=total.dtype, device=total.device)
                    dist.all_gather_into_tensor(gathered_correct, correct)
                    dist.all_gather_into_tensor(gathered_total, total)
                    if rank == 0:
                        writer.add_scalar('acc/test', (gathered_correct.sum() / gathered_total.sum()).item(), step)

                if rank == 0 and config['input_type'] == 'image' and step == config['eval_interval'] \
                        and config['tasks'] * config['train_shots'] <= 100:
                    # Summarize meta-test images once
                    train_x_summ = rearrange(train_x[0], '(tasks s) c h w -> (s tasks) c h w', tasks=config['tasks'])
                    test_x_summ = rearrange(test_x[0], '(tasks s) c h w -> (s tasks) c h w', tasks=config['tasks'])
                    writer.add_image(
                        'meta-test/train',
                        make_grid(train_x_summ, nrow=config['tasks']), step)
                    writer.add_image(
                        'meta-test/test',
                        make_grid(test_x_summ, nrow=config['tasks']), step)

                if rank == 0 and config['dataset'] == 'casia_comp':
                    _, test_num, c, h, w = test_x_bite.shape
                    completion = ((output['logit'][0] + 1) / 2 * 255).round().to(torch.uint8)
                    completion = rearrange(completion, 'n (c h w) -> n c h w', c=c, h=h, w=w)
                    gt = rearrange(test_y_bite[0], 'n (c h w) -> n c h w', c=c, h=h, w=w)

                    # Summarize qualitative examples of image completion
                    comparison, _ = pack([test_x_bite[0], gt, test_x_bite[0], completion], 'n c * w')
                    comparison = rearrange(comparison, '(t s) c h w -> c (s h) (t w)', t=config['tasks'])
                    writer.add_image('meta-test/completion', comparison, step)

            model.train()

    if rank == 0:
        writer.flush()
        end_time = datetime.now()
        print()
        print(f'Training ended at {end_time}')
        print(f'Elapsed time: {end_time - start_time}')
        with open(path.join(config['log_dir'], 'completed.yaml'), 'a') as f:
            yaml.dump({
                'step': step,
                'end_time': end_time,
            }, f)

    dist.destroy_process_group()


def forward_backward(model, train_x, train_y, test_x, test_y, batch_size, config, evaluate=False):
    # Forward
    output = model(train_x, train_y, test_x, test_y, evaluate=evaluate)
    loss = output['loss']
    loss_mean = loss.mean()
    loss_total = loss_mean
    if 'attn_losses' in output and config['attn_loss'] > 0:
        loss_total = loss_total + config['attn_loss'] * sum(output['attn_losses'])

    # Backward with properly weighted loss
    proportion = train_x.shape[0] / batch_size
    loss_total = loss_total * proportion
    loss_total.backward()

    # Return detached output
    detached_output = {
        'loss': loss.detach(),
        'loss_mean': loss_mean.detach(),
        'proportion': proportion,
    }
    if 'evaluation' in output:
        detached_output['evaluation'] = output['evaluation'].detach()
    if 'attn_losses' in output:
        detached_output['attn_losses'] = [attn_loss.detach() for attn_loss in output['attn_losses']]
    if 'inner_lr' in output:
        detached_output['inner_lr'] = output['inner_lr'].detach()

    return detached_output


def forward(model, train_x, train_y, test_x, test_y, eval_size):
    # Forward
    with torch.no_grad():
        output = model(train_x, train_y, test_x, test_y, evaluate=True)
        loss = output['loss']
        loss_mean = loss.mean()

    # Return evaluation output
    eval_output = {
        'loss_mean': loss_mean,
        'proportion': train_x.shape[0] / eval_size,
    }
    if 'logit' in output:
        eval_output['logit'] = output['logit']
    if 'evaluation' in output:
        eval_output['correct'] = output['evaluation'].sum()
        eval_output['total'] = torch.tensor(output['evaluation'].numel(), device=eval_output['correct'].device)

    return eval_output


if __name__ == '__main__':
    main()
