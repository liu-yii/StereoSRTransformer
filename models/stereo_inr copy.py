import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord
from models.crossattention import CAT
from .positionencoder import PositionEncoder


@register('stereoinr')
class StereoINR(nn.Module):
    def __init__(self, encoder_spec, imnet_spec=None,dispnet_spec=None, hidden_dim=128, pb_spec = None, pe_spec = None):
        super().__init__()    
        self.hidden_dim = hidden_dim

        # self.grad_nopad = Get_gradient_nopadding()
        self.encoder = models.make(encoder_spec)
        self.coef = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.cat = CAT(feature_size=[30,90], proj_dim=166)

        self.phase = nn.Linear(2, hidden_dim, bias=False)
        self.imnet = models.make(imnet_spec, args={'in_dim': hidden_dim})
        self.pe = PositionEncoder(posenc_type='sinusoid',complex_transform=False, enc_dims=hidden_dim, hidden_dims=hidden_dim//2)

    def gen_feat(self, inp):
        self.inp_l, self.inp_r = inp.chunk(2, dim=1)
        x = torch.cat((self.inp_l, self.inp_r), dim=0)
        self.feat_coord = make_coord(self.inp_l.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(self.inp_l.shape[0], 2, *self.inp_l.shape[-2:])
        feat_left, feat_right, attn_weight = self.encoder(x)
        disp = self.cat(attn_weight, feat_left, feat_right)
        self.M_right_to_left, self.M_left_to_right = attn_weight

        self.feat_left = feat_left
        self.feat_right = feat_right
        
        return feat_left, feat_right
    
    
    
    def query_rgb(self, lr:Tensor, feat: Tensor, coord: Tensor, cell: Tensor = None):
        """
        Query RGB image based on relative position shift

        :param lr: left or right image, [N,C,H,W]
        :param feat: feature map, [N,C,H,W]
        :param coord: relative position shift, [N,Q,2]
        :param cell: relative cell shift, [N,Q,2]
        """
        coef = self.coef(feat)
        freq = self.freq(feat)

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6 

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                # prepare coefficient & frequency
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_coef = F.grid_sample(
                    coef, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_freq = F.grid_sample(
                    freq, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                
                # prepare cell
                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]
                
                bs, q = coord.shape[:2]
                coord_ff, _ = self.pe(rel_coord)
                q_freq = torch.mul(q_freq, coord_ff)
                q_freq += self.phase(rel_cell.view((bs * q, -1))).view(bs, q, -1)

                inp = torch.mul(q_coef, q_freq)       

                pred = self.imnet(inp.contiguous().view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t
        
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        rgb = ret + F.grid_sample(lr, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        return rgb

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        out_l = self.query_rgb(self.inp_l, self.feat_left, coord, cell)
        out_r = self.query_rgb(self.inp_r, self.feat_right, coord, cell)
        out = torch.cat((out_l, out_r), dim=-1)
        return out