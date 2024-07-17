import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord, warp_grid
from models.crossarch import dispnet, dispnet_norefine
from models.positionencoder import PositionEncoder, PositionEncodingSine1DRelative


@register('stereoinr')
class StereoINR(nn.Module):
    def __init__(self, encoder_spec, imnet_spec=None,dispnet_spec=None, hidden_dim=256, pb_spec = None, pe_spec = None):
        super().__init__()    
        self.hidden_dim = hidden_dim

        # self.grad_nopad = Get_gradient_nopadding()
        self.encoder = models.make(encoder_spec)
        self.coef = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.corr_refine = dispnet(hidden_dim = self.encoder.out_dim, nhead = 4, num_attn_layers = 2)
        # self.corr_refine = dispnet_norefine(hidden_dim = self.encoder.out_dim, nhead = 4, num_attn_layers = 4)
        self.pos_encoder = PositionEncodingSine1DRelative(self.encoder.out_dim, normalize=False)
        self.phase = nn.Linear(2, hidden_dim, bias=False)
        self.imnet = models.make(imnet_spec, args={'in_dim': hidden_dim, 'out_dim': 3})
        self.disp_imnet = models.make(imnet_spec, args={'in_dim': self.encoder.out_dim + 2 + 1, 'out_dim': 1})
        self.pe = PositionEncoder(posenc_type='sinusoid',complex_transform=False, enc_dims=hidden_dim, hidden_dims=hidden_dim//2)

    def gen_feat(self, inp):
        inp_l, inp_r = inp.chunk(2, dim=1)
        x = torch.cat((inp_l, inp_r), dim=0)
        feat_left, feat_right, attn_weight = self.encoder(x)
        M_right_to_left, _ = attn_weight
        return feat_left, feat_right, M_right_to_left
    
    def refine_corr(self, feat_l, feat_r, corr): 
        pos_enc = self.pos_encoder(feat_l)  # [2w-1, c]
        raw_disp, out_left, out_right = self.corr_refine(corr, feat_l, feat_r, pos_enc)
        self.raw_disp = raw_disp
        return raw_disp, out_left, out_right

    
    def query_rgb(self, lr, feat, coord, cell = None, raw_disp = None):
        """
        Query RGB image based on relative position shift

        :param lr: left or right image, [N,C,H,W]
        :param feat: feature map, [N,C,H,W]
        :param raw_disp: disparity map, [N,1,H,W]
        :param coord: relative position shift, [N,Q,2]
        :param cell: relative cell shift, [N,Q,2]
        """
        coef = self.coef(feat)
        freq = self.freq(feat)
        if raw_disp is not None:
            scale = 2/(cell + 1e-6)[:,0,1]/feat.shape[-1]
            scale = scale[:,None,None,None]
            disp = raw_disp * scale
            # normalize disparity
            eps = 1e-6
            mean_disp = torch.mean(disp, dim=(1,2,3), keepdim=True)
            std_disp = torch.std(disp, dim=(1,2,3), keepdim=True) + eps
            disp_normalized = (disp - mean_disp) / std_disp

            pred_disps = []

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
                if raw_disp is not None:
                    # disp_imnet input
                    q_disp = F.grid_sample(
                        disp_normalized, coord_.flip(-1).unsqueeze(1),
                        mode='nearest', align_corners=False)[:, :, 0, :] \
                        .permute(0, 2, 1)
                    q_inp = F.grid_sample(
                        feat, coord_.flip(-1).unsqueeze(1),
                        mode='nearest', align_corners=False)[:, :, 0, :] \
                        .permute(0, 2, 1)
                # imnet input
                q_coef = F.grid_sample(
                    coef, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_freq = F.grid_sample(
                    freq, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                # rel coord
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
                if raw_disp is not None:
                    # pred HR disp
                    # q_disp = torch.mul(q_disp, 2/(rel_cell + 1e-6)[:, :, 1].unsqueeze(-1))
                    disp_inp = torch.cat((q_disp, q_inp, rel_coord), dim=-1)
                    pred_disp = self.disp_imnet(disp_inp).view(bs, q, -1)
                    pred_disps.append(pred_disp)

                # pred HR rgb
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
        
        if raw_disp is not None:
            ret_disp = 0
            for pred_disp, area in zip(pred_disps, areas):
                ret_disp = ret_disp + pred_disp * (area / tot_area).unsqueeze(-1)
            disp = ret_disp + F.grid_sample(self.raw_disp*scale, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        else:
            disp = None        
        return rgb, disp

    
    def warp(self, inp, feat, disp, coord, cell):
        warp_coord = warp_grid(coord, disp, cell) 
        warp_r, _ = self.query_rgb(inp, feat, warp_coord, cell)
        return warp_r
    
    def forward(self, inp, coord, cell):
        out = {}
        inp_l, inp_r = inp.chunk(2, dim=1)
        feat_left, feat_right, M_right_to_left = self.gen_feat(inp)
        raw_disp, out_left, out_right = self.refine_corr(feat_left, feat_right, M_right_to_left)
        # self.feat_left, self.feat_right = out_left, out_right

        out_l, disp = self.query_rgb(inp_l, out_left, coord, cell, raw_disp)
        out_r, _ = self.query_rgb(inp_r, out_right, coord, cell)

        out_rgb = torch.cat((out_l, out_r), dim=-1)
        out['out_rgb'] = out_rgb
        out['raw_disp'] = raw_disp
        # warp for disparity loss (training)
        if self.training:
            with torch.no_grad():
                warp_r = self.warp(inp_l, out_left, disp, coord, cell)
            out['warp_r'] = warp_r
        else:
            out['disp'] = disp
        return out