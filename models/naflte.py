import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord
# from .crossattention_arch import CrossScaleAttention
from .arch_util import PositionalEncoding
from .nafnet import CALayer, RDB, LayerNorm2d

import numpy as np


@register('naflte')
class NAFLTE(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None, hidden_dim=128,pb_spec = None, pe_spec = None):
        super().__init__()   
        self.non_local_attn = True     
        self.multi_scale = [2]
        self.head = 8
        self.hidden_dim = hidden_dim
        self.softmax_scale = 1
        self.feat_unfold = True
        self.local_attn = True
        # self.grad_nopad = Get_gradient_nopadding()
        self.encoder = models.make(encoder_spec)
        self.projl = nn.Conv2d(self.encoder.out_dim, self.encoder.out_dim, 1, 1, 0)
        self.projr = nn.Conv2d(self.encoder.out_dim, self.encoder.out_dim, 1, 1, 0)
        self.norm_l = LayerNorm2d(self.encoder.out_dim)
        self.norm_r = LayerNorm2d(self.encoder.out_dim)
        self.coef = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.phase = nn.Linear(2, hidden_dim//2, bias=False) 
        
        # self.resblock = RDB(G0=self.encoder.out_dim, C=6, G=24)
        # self.CALayer = CALayer(self.encoder.out_dim, self.encoder.out_dim//8)
        

        
        # if self.non_local_attn:
        #     self.cs_attn = CrossScaleAttention(channel=self.hidden_dim, scale=self.multi_scale)
        self.conv_v = nn.Conv2d(self.encoder.out_dim, self.hidden_dim, kernel_size=3, padding=1)
        if self.local_attn:
            self.conv_q = nn.Conv2d(self.encoder.out_dim, hidden_dim, kernel_size=3, padding=1)
            self.conv_k = nn.Conv2d(self.encoder.out_dim, hidden_dim, kernel_size=3, padding=1)
            self.is_pb = True if pb_spec else False
            if self.is_pb:
                self.pb_encoder = models.make(pb_spec, args={'head': self.head}).cuda()
            self.r = 1
        else:
            self.r = 0
        self.r_area = (2 * self.r + 1)**2
        if pe_spec:
            self.pe_encoder = models.make(pe_spec).cuda()
        self.imnet = models.make(imnet_spec, args={'in_dim': hidden_dim})
    
    
    def gen_feat(self, x_left, x_right):
        self.inp_l, self.inp_r = x_left, x_right
        b,c,h,w = self.inp_l.shape
        self.feat_coord = make_coord(self.inp_l.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(self.inp_l.shape[0], 2, *self.inp_l.shape[-2:])
        self.cost = [
            torch.zeros(b, h, w, w).to(x_left.device),
            torch.zeros(b, h, w, w).to(x_left.device)
        ]
        # self.feat_left, self.feat_right, self.disp1, self.disp2, \
        #     (self.M_right_to_left, self.M_left_to_right), (self.V_left, self.V_right) \
        #         = self.encoder(self.inp_l, self.inp_r, self.cost)
        self.feat_left, self.feat_right, self.M_left_to_right, self.M_right_to_left \
            = self.encoder(self.inp_l, self.inp_r, self.cost)
        return self.feat_left, self.feat_right, self.M_left_to_right, self.M_right_to_left
    


    def query_rgb_left(self, coord, cell=None):
        
        # feat_leftW = dispwarpfeature(self.feat_right, self.disp1)
        # left_att = self.att1(self.feat_left)
        # leftW_att = self.att2(feat_leftW)
        # corrleft = (torch.tanh(5*torch.sum(left_att*leftW_att, 1).unsqueeze(1))+1)/2
        # err = self.resblock((feat_leftW - self.feat_left)*corrleft)
        # feat = self.CALayer(err + self.feat_left) #high resolution feature that contains high resolution information of the other image through high res stereo matching
        feat_leftW = torch.matmul(self.M_right_to_left ,self.projr(self.feat_right).permute(0, 2, 3, 1))
        feat = self.feat_left + feat_leftW.permute(0, 3, 1, 2)
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
                
                # basis generation
                bs, q = coord.shape[:2]
                q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)
                q_freq = torch.mul(q_freq, rel_coord.unsqueeze(-1))
                q_freq = torch.sum(q_freq, dim=-2)
                q_freq += self.phase(rel_cell.view((bs * q, -1))).view(bs, q, -1)
                q_freq = torch.cat((torch.cos(np.pi*q_freq), torch.sin(np.pi*q_freq)), dim=-1)

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
        ret += F.grid_sample(self.inp_l, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        return ret
    
    def query_rgb_right(self, coord, cell=None):
        
        # feat_rightW = dispwarpfeature(self.feat_left, self.disp2)
        # right_att = self.att1(self.feat_right)
        # rightW_att = self.att2(feat_rightW)
        # corrright = (torch.tanh(5*torch.sum(right_att*rightW_att, 1).unsqueeze(1))+1)/2
        # err = self.resblock((feat_rightW - self.feat_right)*corrright)
        # feat = self.CALayer(err + self.feat_right) #high resolution feature that contains high resolution information of the other image through high res stereo matching
        feat_rightW = torch.matmul(self.M_left_to_right ,self.projl(self.feat_left).permute(0, 2, 3, 1))
        feat = self.feat_right + feat_rightW.permute(0, 3, 1, 2)
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
                
                # basis generation
                bs, q = coord.shape[:2]
                q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)
                q_freq = torch.mul(q_freq, rel_coord.unsqueeze(-1))
                q_freq = torch.sum(q_freq, dim=-2)
                q_freq += self.phase(rel_cell.view((bs * q, -1))).view(bs, q, -1)
                q_freq = torch.cat((torch.cos(np.pi*q_freq), torch.sin(np.pi*q_freq)), dim=-1)

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
        ret += F.grid_sample(self.inp_r, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        return ret
    

    def forward(self, inp_left, inp_right, coord, cell):
        self.gen_feat(inp_left, inp_right)
        return self.query_rgb_left(coord, cell), self.query_rgb_right(coord, cell), self.M_left_to_right