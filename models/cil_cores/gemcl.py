import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat


class GeMCL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_proj = nn.Linear(256 * (config['x_h'] // 16) * (config['x_w'] // 16), config['hidden_dim'])
        self.ce = nn.CrossEntropyLoss(reduction='none')

        self.map = config['map']

        self.alpha = nn.Parameter(torch.tensor(100.), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(1000.), requires_grad=True)

    def get_prototypes(self, train_x_enc):
        train_x_enc = rearrange(train_x_enc, 'b (t s) h -> b t s h', t=self.config['tasks'])
        prototypes = reduce(train_x_enc, 'b t s h -> b t h', 'mean')

        squared_diff = (train_x_enc - rearrange(prototypes, 'b t h -> b t 1 h')).square()
        squared_diff = reduce(squared_diff, 'b t s h -> b t h', 'sum')

        alpha_prime = self.alpha + self.config['train_shots'] / 2
        beta_prime = self.beta + squared_diff / 2

        if self.map:
            vars = beta_prime / (alpha_prime - 0.5)
        else:
            vars = beta_prime / alpha_prime * (1 + 1 / self.config['train_shots'])

        return prototypes, vars, alpha_prime, beta_prime

    def forward(self, train_x_enc, train_y, test_x_enc, test_y, evaluate):
        batch, test_num = test_y.shape

        # Project image
        train_x_enc = self.image_proj(train_x_enc)
        test_x_enc = self.image_proj(test_x_enc)

        # update prototypes
        prototypes, vars, alpha_prime, beta_prime = self.get_prototypes(train_x_enc)

        # compute logit
        test_x_enc = rearrange(test_x_enc, 'b l h -> b l 1 h')
        prototypes = rearrange(prototypes, 'b t h -> b 1 t h')
        squared_diff = (test_x_enc - prototypes).square()  # b l t h
        vars = repeat(vars, 'b t h -> b l t h', l=test_num)

        eps = 1e-8

        if self.map:
            neg_log_lik = squared_diff / vars
            neg_log_lik = neg_log_lik + (vars+eps).log()
            neg_log_lik = reduce(neg_log_lik, 'b l t h -> b l t', 'sum')
            neg_log_lik = neg_log_lik / 2
        else:
            neg_log_lik = (squared_diff / (vars * alpha_prime * 2) + 1).log()
            neg_log_lik = neg_log_lik * (alpha_prime + 0.5)
            neg_log_lik = neg_log_lik + (vars+eps).log()
            neg_log_lik = reduce(neg_log_lik, 'b l t h -> b l t', 'sum')

        loss = self.ce(rearrange(-neg_log_lik, 'b l t -> (b l) t'), rearrange(test_y, 'b l -> (b l)'))

        output = {
            'loss': loss,
            'logit': -neg_log_lik.detach(),
        }
        if not evaluate:
            return output

        pred = torch.argmin(neg_log_lik, dim=-1)
        evaluation = pred == test_y
        output['evaluation'] = evaluation
        return output