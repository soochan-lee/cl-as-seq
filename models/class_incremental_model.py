import os

import torch
from einops import rearrange, pack

from .cil_cores import CIL_CORE
from .encoders import X_ENCODER
from .model import Model


class ClassIncrementalModel(Model):
    def __init__(self, config):
        super().__init__(config)
        self.x_encoder = X_ENCODER[config['x_encoder']](config)
        # Class Incremental Learning Core
        self.cil_core = CIL_CORE[config['cil_core']](config)

    def forward(self, train_x, train_y, test_x, test_y, evaluate=False):
        # Encode images
        batch, train_num = train_y.shape
        batch, test_num = test_y.shape

        # Encode images
        x, _ = pack([
            rearrange(train_x, 'b l c h w -> (b l) c h w'),
            rearrange(test_x, 'b l c h w -> (b l) c h w'),
        ], '* c h w')
        x_enc = self.x_encoder(x)
        x_enc = rearrange(x_enc, 'bl c h w -> bl (c h w)')
        train_x_enc = rearrange(x_enc[:batch * train_num], '(b l) h -> b l h', b=batch, l=train_num)
        test_x_enc = rearrange(x_enc[batch * train_num:], '(b l) h -> b l h', b=batch, l=test_num)

        return self.cil_core(train_x_enc, train_y, test_x_enc, test_y, evaluate=evaluate)
