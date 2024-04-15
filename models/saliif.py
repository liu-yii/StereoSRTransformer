import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord
from .crossattention_arch import CrossScaleAttention
from .arch_util import PositionalEncoding

import numpy as np

class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0], 
                    [0, 0, 0], 
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0], 
                    [-1, 0, 1], 
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=2)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=2)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=2)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=2)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=2)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=2)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x

class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0], 
                    [0, 0, 0], 
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0], 
                    [-1, 0, 1], 
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding = 1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding = 1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding = 1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding = 1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding = 1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding = 1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x



@register('saliif')
class SALIIF(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None, hidden_dim=128,pb_spec = None, pe_spec = None):
        super().__init__()   
        self.non_local_attn = True     
        self.multi_scale = [2]
        self.head = 8
        self.hidden_dim = hidden_dim
        self.softmax_scale = 1
        self.feat_unfold = True
        self.local_attn = True
        self.grad_nopad = Get_gradient_nopadding()
        self.encoder = models.make(encoder_spec)
        self.coef = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.phase = nn.Linear(2, hidden_dim//2, bias=False)   
        

        
        if self.non_local_attn:
            self.cs_attn = CrossScaleAttention(channel=self.hidden_dim, scale=self.multi_scale)
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
        self.imnet = models.make(imnet_spec, args={'in_dim': self.r_area*self.hidden_dim*3})
    
    
    def gen_feat(self, inp):
        self.inp_l, self.inp_r = torch.chunk(inp, 2,dim=-1)
        self.feat_coord = make_coord(self.inp_l.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(self.inp_l.shape[0], 2, *self.inp_l.shape[-2:])
        
        self.feat = self.encoder(inp)[0]
        # self.grad_feat = self.encoder(self.grad_nopad(inp))
        self.coeff = self.coef(self.feat)
        self.freqq = self.freq(self.feat)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat
        B,C,H,W = feat.shape
        coef = self.coeff
        freq = self.freqq

        coord = coord.unsqueeze(2)


        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6 

        # feature unfold                                //unfold层能不能减少参数？或者用另外的结构替换？
        # if self.feat_unfold:
        #     feat_q = F.unfold(feat, 3, padding=1).view(B, C*9, H, W)         #[16, 576, 48, 48]
        #     feat_k = F.unfold(feat, 3, padding=1).view(B, C*9, H, W)         #[16, 576, 48, 48]
        #     if self.non_local_attn:
        #         non_local_feat_v = self.cs_attn(feat)                        #[16, 64, 48, 48]
        #         feat_v = F.unfold(feat, 3, padding=1).view(B, C*9, H, W)     #[16, 576, 48, 48]
        #         feat_v = torch.cat([feat_v, non_local_feat_v], dim=1)           #[16, 576+64, 48, 48]
        #     else:
        #         feat_v = F.unfold(feat, 3, padding=1).view(B, C*9, H, W)     #[16, 576, 48, 48]
        # else:
        #     feat_q = feat_k = feat_v = feat
        feat_q = self.conv_q(feat)
        feat_k = self.conv_k(feat)
        feat_v = self.conv_v(feat)
        non_local_feat_v = self.cs_attn(feat_v)
        feat_v = torch.cat([feat_v, non_local_feat_v], dim=1)

        query = F.grid_sample(feat_q, coord.flip(-1), mode='nearest', 
                    align_corners=False).permute(0, 3, 2, 1).contiguous()       #[16, 2304, 1, 576]
        

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = self.feat_coord

        r = self.r
        if self.local_attn:
            dh = torch.linspace(-r, r, 2 * r + 1).cuda() * 2 / feat.shape[-2]
            dw = torch.linspace(-r, r, 2 * r + 1).cuda() * 2 / feat.shape[-1]
            # 1, 1, r_area, 2
            delta = torch.stack(torch.meshgrid(dh, dw), axis=-1).view(1, 1, -1, 2)

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                # prepare coefficient & frequency
                coord_ = coord.clone()
                bs, q = coord.shape[:2]
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                
                q_coef = F.grid_sample(                             #[32,2304,256]
                    coef, coord_.flip(-1),
                    mode='nearest', align_corners=False) \
                    .permute(0, 2, 3, 1)
                q_freq = F.grid_sample(                             #[32,2304,256]
                    freq, coord_.flip(-1),
                    mode='nearest', align_corners=False) \
                    .permute(0, 2, 3, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1),
                    mode='nearest', align_corners=False) \
                    .permute(0, 2, 3, 1)
                # local ensemble
                ensamble_coord = coord - q_coord
                ensamble_coord[:, :, :, 0] *= feat.shape[-2]
                ensamble_coord[:, :, :, 1] *= feat.shape[-1]
                area = torch.abs(ensamble_coord[:, :, 0, 0] * ensamble_coord[:, :, 0, 1])
                areas.append(area + 1e-9)
                if self.local_attn:
                    q_coord = q_coord.view(bs, q, 1, -1) + delta

                rel_coord = coord - q_coord
                rel_coord[:, :, :, 0] *= feat.shape[-2]
                rel_coord[:, :, :, 1] *= feat.shape[-1]                                     #[32,2304,2]
                
                # prepare cell
                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]                                     #[32,2304,2]
                
                # 结构学习
                # basis generation
                q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)
                q_freq = torch.mul(q_freq, rel_coord.unsqueeze(-1))
                q_freq = torch.sum(q_freq, dim=-2)
                q_freq = q_freq + self.phase(rel_cell.view((bs * q, -1))).view(bs, q, -1).unsqueeze(2)
                q_freq = torch.cat((torch.cos(np.pi*q_freq), torch.sin(np.pi*q_freq)), dim=-1)
                x1 = torch.mul(q_coef, q_freq)

                # 特征学习
                if self.local_attn:
                    query = query.reshape(bs, q, 1, self.head,
                                            self.hidden_dim // self.head).permute(0, 1, 3, 2, 4)
                     # key and value
                    key = F.grid_sample(feat_k, q_coord.flip(-1), mode='nearest', 
                        align_corners=False).permute(0, 2, 3, 1)          #[16, 2304, 576]
                    key = key.reshape(bs, q, self.r_area, self.head,
                                            self.hidden_dim // self.head).permute(0, 1, 3, 4, 2)
                    
                    value = F.grid_sample(feat_v, q_coord.flip(-1), mode='nearest', align_corners=False).permute(0, 2, 3, 1)         #[16, 2304, 576]
                    
                    # b, q, h, 1, r_area -> b, q, r_area, h
                    similarity = torch.matmul(query, key).reshape(
                        bs, q, self.head, self.r_area
                    ).permute(0, 1, 3, 2) / np.sqrt(self.hidden_dim // self.head)
                    if self.is_pb:
                        _, pb = self.pb_encoder(rel_coord)
                        attn = F.softmax(similarity + pb, dim=-2)
                    else:
                        attn = F.softmax(similarity, dim=-2)
                    attn = attn.reshape(bs, q, self.r_area, self.head, 1)
                    value = value.reshape(bs, q, self.r_area, self.head, self.hidden_dim*2 // self.head)
                    x2 = torch.mul(value, attn).reshape(bs, q, self.r_area, -1)
                    attn_map = attn[0, 10, :, 0, :].reshape(2 * r + 1, 2 * r + 1, 1)
                
                inp = torch.cat([x1, x2], dim=3)         

                pred = self.imnet(inp.contiguous().view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)


        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t
        
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        ret += F.grid_sample(self.inp, coord.flip(-1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, :, 0] \
                      .permute(0, 2, 1)
        return ret

    def forward(self, inp, coord, cell):
        
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)