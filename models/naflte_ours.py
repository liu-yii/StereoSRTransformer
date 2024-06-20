import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

import models
from models import register
from .arch_util import ResidualBlockwithBN
from utils import make_coord, NestedTensor, batched_index_select, torch_1d_sample
from .crossattention_arch import PCAM
# from .arch_util import PositionalEncoding
from .positionencoder import PositionEncoder
from .nafnet import CALayer, RDB, LayerNorm2d
from skimage import morphology
from torch.nn.utils import weight_norm

import numpy as np
import math

@register('naflte_ours')
class NAFLTEOURS(nn.Module):
    def __init__(self, encoder_spec, imnet_spec=None,dispnet_spec=None, hidden_dim=128, pb_spec = None, pe_spec = None):
        super().__init__()    
        self.hidden_dim = hidden_dim

        # self.grad_nopad = Get_gradient_nopadding()
        self.encoder = models.make(encoder_spec)
        self.projl = nn.Conv2d(self.encoder.out_dim, self.encoder.out_dim, 1, 1, 0)
        self.projr = nn.Conv2d(self.encoder.out_dim, self.encoder.out_dim, 1, 1, 0)

        self.coef = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        # dispnet
        self.pcam = PCAM(self.encoder.out_dim)
        self.conv0 = nn.Sequential(weight_norm(nn.Conv2d(1, hidden_dim // 4, 3, 1, 1)),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    weight_norm(nn.Conv2d(hidden_dim // 4, hidden_dim // 2, 1, 1, 0)),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    weight_norm(nn.Conv2d(hidden_dim // 2, hidden_dim, 3, 1, 1)))
        self.query = nn.Conv2d(hidden_dim + self.encoder.out_dim, hidden_dim, 1, 1, 0, bias=True)
        self.key = nn.Conv2d(hidden_dim + self.encoder.out_dim, hidden_dim, 1, 1, 0, bias=True)

        self.phase = nn.Linear(2, hidden_dim, bias=False)
        self.imnet = models.make(imnet_spec, args={'in_dim': hidden_dim})
        self.dispnet = models.make(dispnet_spec, args={'in_dim': hidden_dim})
        self.pe = PositionEncoder(posenc_type='sinusoid',complex_transform=False, enc_dims=hidden_dim, hidden_dims=hidden_dim//2)
        

    def gen_feat(self, x_left, x_right, scale):
        self.inp_l, self.inp_r = x_left, x_right
        self.upscale_factor = scale
        b,c,h,w = self.inp_l.shape
        x = torch.cat((self.inp_l, self.inp_r), dim=0)
        self.feat_coord = make_coord(self.inp_l.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(self.inp_l.shape[0], 2, *self.inp_l.shape[-2:])
        feat_left, feat_right, attn_weight = self.encoder(x)
        
        self.M_right_to_left, self.M_left_to_right = attn_weight

        corelation = attn_weight[0].clone()
        disp, occ = self.pcam(feat_left, feat_right, corelation)

        self.scale = scale[:,None,None,None].repeat(1,1,h,w)
        self.disp = disp
        self.feat_left = feat_left
        self.feat_right = feat_right

        return feat_left, feat_right, disp
    
    def query_rgb_left(self, coord, cell=None):
        feat_leftW = torch.matmul(torch.softmax(self.M_right_to_left, dim=-1), self.projr(self.feat_right).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        feat = self.feat_left + feat_leftW

        coef = self.coef(feat)
        freq = self.freq(feat)

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6 

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = self.feat_coord

        preds = []
        # disps = []
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
        rgb = ret + F.grid_sample(self.inp_l, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        

        return rgb
    
    def query_rgb_right(self, coord, cell=None):
        feat_rightW = torch.matmul(torch.softmax(self.M_left_to_right, dim=-1), self.projr(self.feat_left).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        feat = self.feat_right + feat_rightW
        # key pos
        coef = self.coef(feat)
        freq = self.freq(feat)

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6 

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = self.feat_coord      #16,2,48,48

        preds = []
        # disps = []
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

        rgb = ret + F.grid_sample(self.inp_r, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        
        return rgb
    
    def query_disp(self, coord, cell=None):
        disp_pred = self.disp
        # eps = 1e-6
        # mean_disp_pred = disp_pred.mean()
        # std_disp_pred = disp_pred.std() + eps
        # disp_pred_normalized = self.norm(disp_pred)
        feat = self.conv0(disp_pred)
        feat = torch.cat((feat, self.feat_left), dim=1)
        query = self.query(feat)
        key = self.key(feat)     
        
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6 

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = self.feat_coord      #16,2,48,48

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                # prepare coefficient & frequency
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_freq = F.grid_sample(
                    query, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coef = F.grid_sample(
                    key, coord_.flip(-1).unsqueeze(1),
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

                inp = torch.mul(q_freq, q_coef)
                pred = self.dispnet(inp.contiguous().view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t
        
        ret = 0

        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)

        disp_hr = ret + F.grid_sample(self.disp * self.scale, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        
        return disp_hr
    

    def forward(self, inp_left, inp_right, coord, cell, scale):
        self.gen_feat(inp_left, inp_right, scale)
        return self.query_rgb_left(coord, cell), self.query_rgb_right(coord, cell),\
              self.query_disp(coord, cell)