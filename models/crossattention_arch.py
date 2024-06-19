#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from .ssrtr import LayerNorm2d

class PCAM(nn.Module):
    '''
    Parallax Cross Attention Module (SCAM)
    '''
    def __init__(self, c):
        super().__init__()
        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.occ_head = nn.Sequential(
            weight_norm(nn.Conv2d(1,c, kernel_size=3, padding=1)),
            weight_norm(nn.Conv2d(c,c, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            weight_norm(nn.Conv2d(c, c, kernel_size=3, padding=1)),
            weight_norm(nn.Conv2d(c, c, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.phi = nn.Parameter(torch.tensor(0.0, requires_grad=True))  # dustbin cost
        # self.stereonet = StereoNet(num_feats=c, maxdisp=96)

    def _softmax(self, attn):
        """
        Alternative to optimal transport

        :param attn: raw attention weight, [N,H,W,W]
        :return: updated attention weight, [N,H,W+1,W+1]
        """
        bs, h, w, _ = attn.shape

        # add dustbins
        similarity_matrix = torch.cat([attn, self.phi.expand(bs, h, w, 1).to(attn.device)], -1)
        similarity_matrix = torch.cat([similarity_matrix, self.phi.expand(bs, h, 1, w + 1).to(attn.device)], -2)

        attn_softmax = F.softmax(similarity_matrix, dim=-1)

        return attn_softmax

    def gen_pos_shift(self, w, device):
        """
        Compute relative difference between each pixel location from left image to right image, to be used to calculate
        disparity

        :param w: image width
        :param device: torch device
        :return: relative pos shifts
        """
        pos_r = torch.linspace(0, w - 1, w)[None, None, None, :].to(device)  # 1 x 1 x 1 x W_right
        pos_l = torch.linspace(0, w - 1, w)[None, None, :, None].to(device)  # 1 x 1 x W_left x1
        pos = pos_l - pos_r
        pos[pos < 0] = 0
        return pos
            
    def gen_raw_disp(self, attn_weight, pos_shift, occ_mask = None):
        # b, c, h, w = size
        # find high response area
        high_response = torch.argmax(attn_weight, dim=-1)  # NxHxW

        # build 3 px local window
        response_range = torch.stack([high_response - 1, high_response, high_response + 1], dim=-1)  # NxHxWx3

        # attention with re-weighting
        attn_weight_pad = F.pad(attn_weight, [1, 1], value=0.0)  # N x Hx W_left x (W_right+2)
        attn_weight_rw = torch.gather(attn_weight_pad, -1, response_range + 1)  # offset range by 1, N x H x W_left x 3

        # compute sum of attention
        norm = attn_weight_rw.sum(-1, keepdim=True)
        if occ_mask is None:
            norm[norm < 0.1] = 1.0
        else:
            norm[occ_mask, :] = 1.0  # set occluded region norm to be 1.0 to avoid division by 0

        # re-normalize to 1
        attn_weight_rw = attn_weight_rw / norm  # re-sum to 1
        pos_pad = F.pad(pos_shift, [1, 1]).expand_as(attn_weight_pad)
        pos_rw = torch.gather(pos_pad, -1, response_range + 1)

        # compute low res disparity
        disp_pred_low_res = (attn_weight_rw * pos_rw)  # NxHxW

        return disp_pred_low_res.sum(-1), norm

    def global_correlation_softmax_stereo(self, correlation):
        # global correlation on horizontal direction
        b, h, w, _ = correlation.shape

        x_grid = torch.linspace(0, w - 1, w, device=correlation.device)  # [W]
        
        # mask subsequent positions to make disparity positive
        mask = torch.triu(torch.ones((w, w)), diagonal=1).type_as(correlation)  # [W, W]
        valid_mask = (mask == 0).unsqueeze(0).unsqueeze(0).repeat(b, h, 1, 1)  # [B, H, W, W]
        correlation[~valid_mask] = -1e9

        prob = F.softmax(correlation, dim=-1)  # [B, H, W, W]
        # prob = self._softmax(correlation)[..., :-1, :-1]

        correspondence = (x_grid.view(1, 1, 1, w) * prob).sum(-1)  # [B, H, W]

        # NOTE: unlike flow, disparity is typically positive
        disparity = x_grid.view(1, 1, w).repeat(b, h, 1) - correspondence  # [B, H, W]

        return disparity, prob  # feature resolution

    def forward(self, feat1, feat2, attn_weight):
        b, c, h, w = feat1.shape
        disp, prob = self.global_correlation_softmax_stereo(attn_weight)
        disp = disp.clamp(min=0)  # positive disparity
        
        occ_mask = torch.sum(prob, dim=-1).view(b, 1, h, w)
        occ = self.occ_head(occ_mask)
        
        return disp.unsqueeze(1), occ