import torch
import torch.nn as nn
from einops import rearrange, reduce


class PrototypicalNetwork(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_proj = nn.Linear(256 * (config['x_h'] // 16) * (config['x_w'] // 16), config['hidden_dim'])
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, train_x_enc, train_y, test_x_enc, test_y, evaluate):
        batch, test_num = test_y.shape

        # Project image
        train_x_enc = self.image_proj(train_x_enc)
        test_x_enc = self.image_proj(test_x_enc)

        mask = torch.eq(
            rearrange(train_y, 'b l -> b 1 l'),
            rearrange(torch.arange(self.config['tasks'], device=train_y.device), 't -> 1 t 1')
        ).float()
        num_per_task = reduce(mask, 'b t l -> b t 1', 'sum')  # each task should have at least one sample
        prototypes = torch.einsum('btl,bld->btd', mask, train_x_enc) / num_per_task
        prototypes = rearrange(prototypes, 'b t d -> b 1 t d')
        test_x_enc = rearrange(test_x_enc, 'b n d -> b n 1 d')
        logit = -reduce((test_x_enc - prototypes) ** 2, 'b n t d -> b n t', 'sum')
        loss = self.ce(rearrange(logit, 'b n t -> (b n) t'), rearrange(test_y, 'b n -> (b n)'))
        loss = rearrange(loss, '(b n) -> b n 1', b=batch, n=test_num)

        output = {
            'loss': loss,
            'logit': logit.detach(),
        }
        if not evaluate:
            return output

        pred = torch.argmax(logit, dim=-1)
        evaluation = pred == test_y
        output['evaluation'] = evaluation
        return output
