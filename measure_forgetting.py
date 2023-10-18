import os
import os.path as path
import socket
from argparse import ArgumentParser
from datetime import datetime
from glob import glob

import math
import torch
import yaml
from einops import rearrange, reduce
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from dataset import DATASET
from models import MODEL

parser = ArgumentParser()
parser.add_argument('--model-config', '-mc', required=True)
parser.add_argument('--data-config', '-dc', required=True)
parser.add_argument('--log-dir', '-l')
parser.add_argument('--override', '-o', default='')

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

    assert world_size == 1, 'Multi-GPU is not supported.'

    # Initialize process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    # Build model
    model = MODEL[config['model']](config).to(rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])
    optim = getattr(torch.optim, config['optim'])(model.parameters(), **config['optim_args'])
    lr_sched = getattr(lr_scheduler, config['lr_sched'])(optim, **config['lr_sched_args'])

    # Resume checkpoint
    ckpt_paths = sorted(glob(path.join(config['log_dir'], 'ckpt-*.pt')))
    if len(ckpt_paths) == 0:
        raise RuntimeError(f'No checkpoint found in {config["log_dir"]}')
    ckpt_path = ckpt_paths[-1]
    # Get step number from checkpoint name
    start_step = int(path.basename(ckpt_path).split('-')[1].split('.')[0])
    if start_step != config['max_train_steps']:
        raise RuntimeError(f'Latest checkpoint {ckpt_path} does not match max_train_steps {config["max_train_steps"]}')
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model'])
    optim.load_state_dict(ckpt['optim'])
    lr_sched.load_state_dict(ckpt['lr_sched'])
    print(f'Checkpoint loaded from {ckpt_path}')
    optim.zero_grad()
    model.eval()

    total_tasks = config['tasks']

    # Data
    Dataset = DATASET[config['dataset']]
    test_set = Dataset(config, root='./data', meta_split='test')

    start_time = datetime.now()
    print(f'Computation started at {start_time}')

    task_interval = 1
    scores = {
        'loss': {},
        'error': {},
        'correct': {},
        'total': {},
    }
    for tasks in tqdm(range(task_interval, total_tasks + 1, task_interval)):
        config['tasks'] = tasks
        test_loader = DataLoader(
            test_set,
            batch_size=config['eval_batch_size'],
            num_workers=config['num_workers'])
        test_loader_iter = iter(test_loader)

        loss_mean = [0.] * tasks
        correct, total = [0] * tasks, [0] * tasks
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

                # Forward
                with torch.no_grad():
                    output = model(train_x_bite, train_y_bite, test_x_bite, test_y_bite, evaluate=True)
                digested += bite

                if 'evaluation' in output:
                    # Classification accuracy
                    evaluation = rearrange(output['evaluation'], 'b (t s) -> b t s', t=tasks, s=config['test_shots'])
                    t_correct = reduce(evaluation, 'b t s -> t', 'sum')
                    t_total = evaluation.shape[0] * evaluation.shape[2]
                    for t in range(tasks):
                        correct[t] += t_correct[t].item()
                        total[t] += t_total

                # Loss
                if len(output['loss'].shape) == 3:
                    loss = reduce(output['loss'], 'b (t s) y -> t', 'mean', t=tasks, s=config['test_shots'])
                elif len(output['loss'].shape) == 2:
                    loss = reduce(output['loss'], 'b (t s) -> t', 'mean', t=tasks, s=config['test_shots'])
                else:
                    raise RuntimeError('Unexpected loss shape')
                for t in range(tasks):
                    loss_mean[t] += loss[t].item() * bite / eval_size

        scores['loss'][tasks] = loss_mean
        if total[0] > 0:
            scores['error'][tasks] = [100. - 100. * c / t for c, t in zip(correct, total)]
            scores['correct'][tasks] = correct
            scores['total'][tasks] = total

    if rank == 0:
        end_time = datetime.now()
        print()
        print(f'Evaluation ended at {end_time}')
        print(f'Elapsed time: {end_time - start_time}')
        torch.save(scores, path.join(config['log_dir'], 'forgetting.pt'))

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
