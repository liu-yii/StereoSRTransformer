import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import numpy as np
from models import register

@register('posenc')
class PositionEncoder(nn.Module):
    def __init__(
        self,
        posenc_type=None,
        complex_transform=False,
        posenc_scale=6,
        gauss_scale=1,
        in_dims=2,
        enc_dims=256,
        hidden_dims=32,
        head=1,
        gamma=1
    ):
        super().__init__()

        self.posenc_type = posenc_type
        self.complex_transform = complex_transform
        self.posenc_scale = posenc_scale
        self.gauss_scale = gauss_scale

        self.in_dims = in_dims
        self.enc_dims = enc_dims
        self.hidden_dims = hidden_dims
        self.head = head
        self.gamma = gamma

        self.define_parameter()

    def define_parameter(self):
        if self.posenc_type == 'sinusoid' or self.posenc_type == 'ipe':
            self.b_vals = 2.**torch.linspace(
                0, self.posenc_scale, self.enc_dims // 4
            ) - 1  # -1 -> (2 * pi)
            self.b_vals = torch.stack([self.b_vals, torch.zeros_like(self.b_vals)], dim=-1)
            self.b_vals = torch.cat([self.b_vals, torch.roll(self.b_vals, 1, -1)], dim=0)
            self.a_vals = torch.ones(self.b_vals.shape[0])
            self.proj = nn.Linear(self.enc_dims, self.head)

        elif self.posenc_type == 'learn':
            self.Wr = nn.Linear(self.in_dims, self.hidden_dims // 2, bias=False)
            self.mlp = nn.Sequential(
                nn.LayerNorm(self.hidden_dims, eps=1e-6),
                nn.Linear(self.hidden_dims, self.hidden_dims),
                nn.GELU(),
                nn.LayerNorm(self.hidden_dims, eps=1e-6),
                nn.Linear(self.hidden_dims, self.enc_dims)
            )
            self.proj = nn.Sequential(nn.GELU(), nn.Linear(self.enc_dims, self.head))
            self.init_weight()

        elif self.posenc_type == 'dpb':
            self.mlp = nn.Sequential(
                nn.Linear(2, self.hidden_dims),
                nn.LayerNorm(self.hidden_dims, eps=1e-6),
                nn.ReLU(),
                nn.Linear(self.hidden_dims, self.hidden_dims),
                nn.LayerNorm(self.hidden_dims, eps=1e-6),
                nn.ReLU(),
                nn.Linear(self.hidden_dims, self.enc_dims)
            )
            self.proj = nn.Sequential(
                nn.LayerNorm(self.hidden_dims, eps=1e-6),
                nn.ReLU(),
                nn.Linear(self.enc_dims, self.head)
            )

    def init_weight(self):
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, positions, cells=None):

        if self.posenc_type is None:
            return positions

        if self.posenc_type == 'sinusoid' or self.posenc_type == 'ipe':
            self.b_vals = self.b_vals.cuda()
            self.a_vals = self.a_vals.cuda()

            # b, q, 1, c (x -> c/2, y -> c/2)
            sin_part = self.a_vals * torch.sin(
                torch.matmul(positions, self.b_vals.transpose(-2, -1))
            )
            cos_part = self.a_vals * torch.cos(
                torch.matmul(positions, self.b_vals.transpose(-2, -1))
            )

            if self.posenc_type == 'ipe':
                # b, q, 2
                cell = cells.clone()
                cell_part = torch.sinc(
                    torch.matmul((1 / np.pi * cell), self.b_vals.transpose(-2, -1))
                )

                sin_part = sin_part * cell_part
                cos_part = cos_part * cell_part

            if self.complex_transform:
                pos_enocoding = torch.view_as_complex(torch.stack([cos_part, sin_part], dim=-1))
            else:
                pos_enocoding = torch.cat([sin_part, cos_part], dim=-1)
                pos_bias = self.proj(pos_enocoding)

        elif self.posenc_type == 'learn':
            projected_pos = self.Wr(positions)

            sin_part = torch.sin(projected_pos)
            cos_part = torch.cos(projected_pos)

            if self.complex_transform:
                pos_enocoding = 1 / np.sqrt(self.hidden_dims) * torch.view_as_complex(
                    torch.stack([cos_part, sin_part], dim=-1)
                )
            else:
                pos_enocoding = 1 / np.sqrt(self.hidden_dims
                                           ) * torch.cat([sin_part, cos_part], dim=-1)
                pos_enocoding = self.mlp(pos_enocoding)

        elif self.posenc_type == 'dpb':
            pos_enocoding = self.mlp(positions)

        pos_bias = None if self.complex_transform else self.proj(pos_enocoding)

        return pos_enocoding, pos_bias
    


class PositionEncodingSine1DRelative(nn.Module):
    """
    relative sine encoding 1D, partially inspired by DETR (https://github.com/facebookresearch/detr)
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    @torch.no_grad()
    def forward(self, input):
        """
        :param inputs: Tensor 
        :return: pos encoding [N,C,H,2W-1]
        """
        x = input

        # update h and w if downsampling
        bs, _, h, w = x.size()

        # populate all possible relative distances
        x_embed = torch.linspace(w - 1, -w + 1, 2 * w - 1, dtype=torch.float32, device=x.device)

        if self.normalize:
            x_embed = x_embed * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t  # 2W-1xC
        pos = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)  # 2W-1xC
        # pos = pos.view(1, -1, 1, 2*w-1).repeat(bs, 1, h, 1)  # NxCxHx2W-1
        
        return pos
