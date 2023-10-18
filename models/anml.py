import math
import torch
from einops import rearrange, repeat, pack
from torch import nn
from torch.nn import functional as F

from models.encoders.classification import ClassEncoder
from utils import cross_entropy, angle_loss
from .model import Model


class PredictionNetworkForImageInput(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.inner_log_lr = nn.Parameter(
            torch.tensor(math.log(config['inner_lr']), dtype=torch.float),
            requires_grad=config['learnable_lr'])

        self.config = config

        if config['output_type'] == 'class':
            output_dim = config['y_vocab']
            self.loss_fn = cross_entropy
        elif config['output_type'] == 'vector':
            output_dim = config['y_dim']
            if config['output_activation'] == 'angle':
                self.loss_fn = angle_loss
            else:
                self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise ValueError('Output type must be class or vector.')

        self.net = nn.Sequential(
            # 32 x 32
            nn.Conv2d(config['x_c'], 32, kernel_size=3, stride=1, padding=1, bias=True),
            # 32 x 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),
            # 16 x 16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
            # 8 x 8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=True),
            # 4 x 4
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),
            # 2 x 2
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * (config['x_h'] // 16) * (config['x_w'] // 16), output_dim)
        )

    def conv2d_batch(self, input, weight, bias=None, stride=1, padding=1):
        '''
        Args:
            input: [batch, in_channel, in_height, in_width]
            weight: [batch, out_channel, in_channel, kernel_height, kernel_width]
            bias: [batch, out_channel]
        Returns:
            output: [batch, out_channel, out_height, out_width]
        '''
        batch, out_channel, in_channel, kernel_height, kernel_width = weight.data.shape
        weight = rearrange(weight, 'b oc ic kh kw -> b oc (ic kh kw)')

        input = F.pad(input, (padding, padding, padding, padding))

        if len(input.shape) == 4:
            batch, in_channel, in_height, in_width = input.shape

            inputs = []
            for i in range(kernel_height):
                for j in range(kernel_width):
                    inputs.append(input[:, :, i:i + in_height - kernel_height + 1:stride,
                                  j:j + in_width - kernel_width + 1:stride])

            input, _ = pack(inputs, 'b * h w')

            output = torch.einsum('boc,bchw->bohw', weight, input)

            if bias is not None:
                output = output + rearrange(bias, 'b o -> b o 1 1')

        elif len(input.shape) == 5:
            batch, test_num, in_channel, in_height, in_width = input.shape

            inputs = []
            for i in range(kernel_height):
                for j in range(kernel_width):
                    inputs.append(input[:, :, :, i:i + in_height - kernel_height + 1:stride,
                                  j:j + in_width - kernel_width + 1:stride])

            input, _ = pack(inputs, 'b l * h w')

            output = torch.einsum('boc,blchw->blohw', weight, input)

            if bias is not None:
                output = output + rearrange(bias, 'b o -> b 1 o 1 1')
        else:
            raise ValueError('Input shape must be 4 or 5.')

        return output

    def forward(self, train_x, train_y, test_x, test_y, train_gate, test_gate, evaluate):
        batch, train_num, x_c, x_h, x_w = train_x.shape
        batch, test_num, x_c, x_h, x_w = test_x.shape

        # Forward training data
        inner_lr = self.inner_log_lr.exp()
        with torch.enable_grad():

            fast_weights = [
                repeat(p, '... -> b ...', b=batch)
                for p in self.net.parameters()
            ]

            for i in range(train_num):
                x = train_x[:, i]

                for j in range(5):
                    stride = 1 if j == 0 else 2
                    x = self.conv2d_batch(
                        x, weight=fast_weights[j * 2], bias=fast_weights[j * 2 + 1], stride=stride, padding=1)
                    x = F.relu(x)

                x = torch.flatten(x, start_dim=1)
                x = x * train_gate[:, i]
                logit = torch.einsum('bnm,bm->bn', fast_weights[-2], x) + fast_weights[-1]

                if self.config['output_type'] == 'class':
                    loss = self.loss_fn(logit, train_y[:, i]).sum()
                elif self.config['output_type'] == 'vector':
                    if self.config['output_activation'] == 'angle':
                        loss = self.loss_fn(logit, train_y[:, i]).sum()
                    elif self.config['output_activation'] == 'tanh':
                        logit = torch.tanh(logit)
                        loss = self.loss_fn(logit, train_y[:, i]).mean(-1).sum()
                    else:
                        loss = self.loss_fn(logit, train_y[:, i]).mean(-1).sum()

                # Update fast weights
                grad = torch.autograd.grad(loss, fast_weights, create_graph=True)
                fast_weights = [
                    old - inner_lr * g
                    for old, g in zip(fast_weights, grad)
                ]

        # Forward test data
        x = test_x
        for j in range(5):
            stride = 1 if j == 0 else 2
            x = self.conv2d_batch(x, weight=fast_weights[j * 2], bias=fast_weights[j * 2 + 1], stride=stride, padding=1)
            x = F.relu(x)

        x = torch.flatten(x, start_dim=2)
        x = x * test_gate
        logit = torch.einsum('bnm,blm->bln', fast_weights[-2], x) + rearrange(fast_weights[-1], 'b n -> b 1 n')

        if self.config['output_type'] == 'class':
            loss = self.loss_fn(logit, test_y)
            loss = rearrange(loss, 'b l -> b l 1', b=batch)

        elif self.config['output_type'] == 'vector':
            if self.config['output_activation'] == 'angle':
                loss = self.loss_fn(logit, test_y)
            elif self.config['output_activation'] == 'tanh':
                logit = torch.tanh(logit)
                loss = self.loss_fn(logit, test_y).mean(-1)
            else:
                loss = self.loss_fn(logit, test_y).mean(-1)
            loss = rearrange(loss, 'b l -> b l 1', b=batch)

        if not evaluate or self.config['output_type'] == 'vector':
            return loss, inner_lr, None, logit

        evaluation = logit.argmax(-1) == test_y
        return loss, inner_lr, evaluation, logit


class FCLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc1 = nn.Parameter(
            torch.FloatTensor(config['hidden_dim'], config['x_dim']),
            requires_grad=True)
        self.b1 = nn.Parameter(torch.FloatTensor(config['hidden_dim']), requires_grad=True)
        self.fc2 = nn.Parameter(
            torch.FloatTensor(config['hidden_dim'], config['hidden_dim']),
            requires_grad=True)
        self.b2 = nn.Parameter(torch.FloatTensor(config['hidden_dim']), requires_grad=True)
        self.fc3 = nn.Parameter(
            torch.FloatTensor(config['hidden_dim'], config['hidden_dim']),
            requires_grad=True)
        self.b3 = nn.Parameter(torch.FloatTensor(config['hidden_dim']), requires_grad=True)
        self.fc4 = nn.Parameter(torch.FloatTensor(config['y_dim'], config['hidden_dim']), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc1)
        nn.init.zeros_(self.b1)
        nn.init.xavier_normal_(self.fc2)
        nn.init.zeros_(self.b2)
        nn.init.xavier_normal_(self.fc3)
        nn.init.zeros_(self.b3)
        nn.init.xavier_normal_(self.fc4)


class PredictionNetworkForVectorInput(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.inner_log_lr = nn.Parameter(
            torch.tensor(math.log(config['inner_lr']), dtype=torch.float),
            requires_grad=config['learnable_lr'])

        self.loss_fn = nn.MSELoss(reduction='none')

        self.net = FCLayer(config)

    def forward(self, train_x, train_y, test_x, test_y, train_gate, test_gate, evaluate):
        batch, train_num, x_dim = train_x.shape

        inner_lr = self.inner_log_lr.exp()
        with torch.enable_grad():
            fast_weights = [
                repeat(p, '... -> b ...', b=batch)
                for p in self.net.parameters()
            ]

            for i in range(train_num):
                # Forward training data
                x = train_x[:, i]
                x = torch.einsum('bnm,bm->bn', fast_weights[0], x) + fast_weights[1]
                x = F.relu(x)
                x = torch.einsum('bnm,bm->bn', fast_weights[2], x) + fast_weights[3]
                x = F.relu(x)
                x = torch.einsum('bnm,bm->bn', fast_weights[4], x) + fast_weights[5]
                x = F.relu(x)

                x = x * train_gate[:, i]
                logit = torch.einsum('bnm,bm->bn', fast_weights[6], x)

                loss = self.loss_fn(logit, train_y[:, i]).mean(-1).sum()

                # Update fast weights
                grad = torch.autograd.grad(loss, fast_weights, create_graph=True)
                fast_weights = [
                    old - inner_lr * g
                    for old, g in zip(fast_weights, grad)
                ]

        x = torch.einsum('bnm,blm->bln', fast_weights[0], test_x) + rearrange(fast_weights[1], 'b m -> b 1 m')
        x = F.relu(x)
        x = torch.einsum('bnm,blm->bln', fast_weights[2], x) + rearrange(fast_weights[3], 'b m -> b 1 m')
        x = F.relu(x)
        x = torch.einsum('bnm,blm->bln', fast_weights[4], x) + rearrange(fast_weights[5], 'b m -> b 1 m')
        x = F.relu(x)

        x = x * test_gate
        logit = torch.einsum('bnm,blm->bln', fast_weights[6], x)

        logit = rearrange(logit, 'b l d -> (b l) d')
        test_y = rearrange(test_y, 'b l d -> (b l) d')
        loss = self.loss_fn(logit, test_y).mean(-1)
        evaluation = loss.detach()
        loss = rearrange(loss, '(b l) -> b l 1', b=batch)

        return loss, inner_lr, evaluation, logit


class ANML(Model):
    def __init__(self, config):
        super().__init__(config)

        if config['input_type'] == 'image':
            self.neuromodulatory_network = nn.Sequential(
                nn.Conv2d(config['x_c'], 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.Sigmoid(),
                nn.Flatten())
            self.prediction_network = PredictionNetworkForImageInput(config)
            self.y_encoder = ClassEncoder(config, use_embedding=False)

        elif config['input_type'] == 'vector':
            self.neuromodulatory_network = nn.Sequential(
                nn.Linear(config['x_dim'], config['hidden_dim'], bias=False),
                nn.BatchNorm1d(config['hidden_dim']),
                nn.ReLU(inplace=True),
                nn.Linear(config['hidden_dim'], config['hidden_dim'],
                          bias=False),
                nn.BatchNorm1d(config['hidden_dim']),
                nn.ReLU(inplace=True),
                nn.Linear(config['hidden_dim'], config['hidden_dim'],
                          bias=False),
                nn.BatchNorm1d(config['hidden_dim']),
                nn.Sigmoid())

            self.prediction_network = PredictionNetworkForVectorInput(config)
            self.y_encoder = None

        else:
            raise ValueError(f'Unknown input type {config["input_type"]}')

    def forward(self, train_x, train_y, test_x, test_y, evaluate=False):
        if self.config['input_type'] == 'image':
            batch, train_num, x_c, x_h, x_w = train_x.shape
            batch, test_num, x_c, x_h, x_w = test_x.shape

            if self.config['output_type'] == 'class':
                # Encode labels
                y_codebook = self.y_encoder.sample_codebook(batch, device=train_y.device)
                train_y = self.y_encoder.y2code(train_y, y_codebook)
                test_y = self.y_encoder.y2code(test_y, y_codebook)
                train_y = rearrange(train_y, 'b l 1 -> b l')
                test_y = rearrange(test_y, 'b l 1 -> b l')
            elif self.config['output_type'] == 'vector':
                if self.config['output_activation'] == 'tanh':
                    train_y = train_y.float() * 2 / 255 - 1
                    test_y = test_y.float() * 2 / 255 - 1
            else:
                raise NotImplementedError

            # normalize x to [-1, 1]
            train_x = train_x.float() * 2 / 255 - 1
            test_x = test_x.float() * 2 / 255 - 1

            x, _ = pack([
                rearrange(train_x, 'b l c h w -> (b l) c h w'),
                rearrange(test_x, 'b l c h w -> (b l) c h w'),
            ], '* c h w')

            train_test_x, _ = pack([train_x, test_x], '* c h w')

            train_test_gate = self.neuromodulatory_network(train_test_x)  # [batch*train_num+batch*test_num, hidden_dim]
            train_gate = rearrange(train_test_gate[:batch * train_num], '(b l) h -> b l h', b=batch, l=train_num)
            test_gate = rearrange(train_test_gate[batch * train_num:], '(b l) h -> b l h', b=batch, l=test_num)

            meta_loss, inner_lr, evaluation, logit = self.prediction_network(
                train_x, train_y, test_x, test_y, train_gate, test_gate, evaluate)

        elif self.config['input_type'] == 'vector':
            batch, train_num, x_dim = train_x.shape
            batch, test_num, y_dim = test_y.shape

            # Encode images
            train_test_x, _ = pack([
                rearrange(train_x, 'b l d -> (b l) d'),
                rearrange(test_x, 'b l d -> (b l) d'),
            ], '* d')

            train_test_gate = self.neuromodulatory_network(train_test_x)  # [batch*train_num+batch*test_num, hidden_dim]
            train_gate = rearrange(train_test_gate[:batch * train_num], '(b l) h -> b l h', b=batch, l=train_num)
            test_gate = rearrange(train_test_gate[batch * train_num:], '(b l) h -> b l h', b=batch, l=test_num)

            meta_loss, inner_lr, evaluation, logit = self.prediction_network(
                train_x, train_y, test_x, test_y, train_gate, test_gate, evaluate)

        else:
            raise ValueError(f'Unknown input type {self.config["input_type"]}')

        output = {
            'loss': meta_loss,
            'logit': logit.detach(),
            'inner_lr': inner_lr.detach(),
        }

        if evaluation is None:
            return output

        output['evaluation'] = evaluation
        return output
