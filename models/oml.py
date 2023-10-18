import math

import torch
from einops import rearrange, repeat, pack
from torch import nn
from torch.nn import functional as F

from models import Model
from models.encoders import X_ENCODER, MlpEncoder
from models.encoders.classification import ClassEncoder
from utils import angle_loss, cross_entropy


class OML(Model):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.inner_log_lr = nn.Parameter(
            torch.tensor(math.log(config['inner_lr']), dtype=torch.float),
            requires_grad=config['learnable_lr'])
        self.input_type = config['input_type']
        self.output_type = config['output_type']

        if self.input_type == 'image':
            self.x_encoder = X_ENCODER[config['x_encoder']](config)
        elif self.input_type == 'vector':
            self.x_encoder = MlpEncoder(config, input_dim=config['x_dim'])
        else:
            raise NotImplementedError

        if self.output_type == 'class':
            assert config['y_len'] == 1
            self.y_encoder = ClassEncoder(config, use_embedding=False)
            self.loss_fn = cross_entropy
        elif self.output_type == 'vector':
            self.y_encoder = None
            if config['output_activation'] == 'angle':
                self.loss_fn = angle_loss
            else:
                self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError

        if 'fast_dim' not in config:
            config['fast_dim'] = config['hidden_dim']
        if 'fast_layers' not in config:
            config['fast_layers'] = 2
        assert config['fast_layers'] >= 2

        # Create fast weights
        self.fast_weights = nn.ParameterList()
        if config['input_type'] == 'image':
            w_in = nn.Parameter(
                torch.FloatTensor(config['fast_dim'], 256 * (config['x_h'] // 16) * (config['x_w'] // 16)),
                requires_grad=True)
        else:
            w_in = nn.Parameter(
                torch.FloatTensor(config['fast_dim'], config['hidden_dim']),
                requires_grad=True)
        b_in = nn.Parameter(torch.FloatTensor(config['fast_dim']), requires_grad=True)
        self.fast_weights.append(w_in)
        self.fast_weights.append(b_in)

        # Additional fast layers
        for i in range(config['fast_layers'] - 2):
            w = nn.Parameter(
                torch.FloatTensor(config['fast_dim'], config['fast_dim']),
                requires_grad=True)
            b = nn.Parameter(torch.FloatTensor(config['fast_dim']), requires_grad=True)
            self.fast_weights.append(w)
            self.fast_weights.append(b)

        # Output fast layer
        if config['output_type'] == 'class':
            w_out = nn.Parameter(torch.FloatTensor(config['y_vocab'], config['fast_dim']), requires_grad=True)
        elif config['output_type'] == 'vector':
            w_out = nn.Parameter(torch.FloatTensor(config['y_dim'], config['fast_dim']), requires_grad=True)
        else:
            raise NotImplementedError
        self.fast_weights.append(w_out)

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.fast_weights:
            if len(p.shape) == 2:
                nn.init.xavier_normal_(p)
            elif len(p.shape) == 1:
                nn.init.zeros_(p)
            else:
                raise NotImplementedError

    def forward(self, train_x, train_y, test_x, test_y, evaluate):
        if self.input_type == 'image':
            batch, train_num, c, h, w = train_x.shape
            batch, test_num, c, h, w = test_x.shape

            # Encode images
            x, _ = pack([
                rearrange(train_x, 'b l c h w -> (b l) c h w'),
                rearrange(test_x, 'b l c h w -> (b l) c h w'),
            ], '* c h w')
            x_enc = self.x_encoder(x)
            x_enc = rearrange(x_enc, 'bl c h w -> bl (c h w)')
            train_x_enc = rearrange(x_enc[:batch * train_num], '(b l) h -> b l h', b=batch, l=train_num)
            test_x_enc = rearrange(x_enc[batch * train_num:], '(b l) h -> b l h', b=batch, l=test_num)
        elif self.input_type == 'vector':
            batch, train_num, x_dim = train_x.shape
            batch, test_num, x_dim = test_x.shape

            # Encode x vectors
            x, _ = pack([
                rearrange(train_x, 'b l d -> (b l) d'),
                rearrange(test_x, 'b l d -> (b l) d'),
            ], '* d')
            x_enc = self.x_encoder(x)
            train_x_enc = rearrange(x_enc[:batch * train_num], '(b l) h -> b l h', b=batch, l=train_num)
            test_x_enc = rearrange(x_enc[batch * train_num:], '(b l) h -> b l h', b=batch, l=test_num)
        else:
            raise NotImplementedError

        if self.output_type == 'class':
            # Encode labels
            y_codebook = self.y_encoder.sample_codebook(batch, device=train_y.device)
            batch, num_classes, y_len = y_codebook.shape
            train_y_code = self.y_encoder.y2code(train_y, y_codebook)
            test_y_code = self.y_encoder.y2code(test_y, y_codebook)
            train_y_code = rearrange(train_y_code, 'b l 1 -> b l')
            test_y_code = rearrange(test_y_code, 'b l 1 -> b l')
        elif self.output_type == 'vector':
            if self.config['output_activation'] == 'tanh':
                assert train_y.dtype == torch.uint8
                train_y = train_y.float() * 2 / 255 - 1
                test_y = test_y.float() * 2 / 255 - 1
        else:
            raise NotImplementedError

        inner_lr = self.inner_log_lr.exp()
        with torch.enable_grad():
            # Copy weights for each sequence in the batch
            fast_weights = [
                repeat(p, '... -> b ...', b=batch)
                for p in self.fast_weights
            ]

            # Inner loop
            for i in range(train_num):
                # Forward training data
                x_i = train_x_enc[:, i]
                logit = fast_forward_train(x_i, fast_weights)
                if self.output_type == 'class':
                    loss = self.loss_fn(logit, train_y_code[:, i]).sum()
                elif self.output_type == 'vector':
                    if self.config['output_activation'] == 'angle':
                        loss = self.loss_fn(logit, train_y[:, i]).sum()
                    elif self.config['output_activation'] == 'tanh':
                        logit = torch.tanh(logit)
                        loss = self.loss_fn(logit, train_y[:, i]).mean(-1).sum()
                    else:
                        loss = self.loss_fn(logit, train_y[:, i]).mean(-1).sum()
                else:
                    raise NotImplementedError

                # Update fast weights
                grad = torch.autograd.grad(loss, fast_weights, create_graph=True)
                fast_weights = [
                    old - inner_lr * g
                    for old, g in zip(fast_weights, grad)
                ]

        # Forward test data
        logit = fast_forward_test(test_x_enc, fast_weights)
        if self.output_type == 'class':
            meta_loss = self.loss_fn(logit, test_y_code)
        elif self.output_type == 'vector':
            if self.config['output_activation'] == 'angle':
                meta_loss = self.loss_fn(logit, test_y)
            elif self.config['output_activation'] == 'tanh':
                logit = torch.tanh(logit)
                meta_loss = self.loss_fn(logit, test_y).mean(-1)
            else:
                meta_loss = self.loss_fn(logit, test_y).mean(-1)
        else:
            raise NotImplementedError
        meta_loss = rearrange(meta_loss, 'b l -> b l 1')

        output = {
            'loss': meta_loss,
            'logit': logit.detach(),
            'inner_lr': inner_lr.detach(),
        }
        if not evaluate:
            return output

        if self.output_type == 'vector':
            return output

        evaluation = logit.argmax(-1) == test_y_code
        output['evaluation'] = evaluation
        return output


def fast_forward_train(x, fast_weights):
    for p in fast_weights:
        if len(p.shape) == 3:
            x = torch.einsum('bnm,bm->bn', p, x)
        elif len(p.shape) == 2:
            x = F.relu(x + p)
        else:
            raise NotImplementedError
    return x


def fast_forward_test(x, fast_weights):
    for p in fast_weights:
        if len(p.shape) == 3:
            x = torch.einsum('bnm,blm->bln', p, x)
        elif len(p.shape) == 2:
            x = F.relu(x + rearrange(p, 'b m -> b 1 m'))
        else:
            raise NotImplementedError
    return x
