import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

import models
from models import register
from utils import make_coord, NestedTensor, batched_index_select, torch_1d_sample
from .crossattention_arch import CrossScaleAttention
# from .arch_util import PositionalEncoding
from .positionencoder import PositionEncoder
from .nafnet import CALayer, RDB, LayerNorm2d
from skimage import morphology

import numpy as np
import math

@register('naflte_ours')
class NAFLTEOURS(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,dispnet_spec=None, hidden_dim=128, pb_spec = None, pe_spec = None):
        super().__init__()   
        self.non_local_attn = True     
        self.multi_scale = [2]
        self.head = 4
        self.window_size = 8
        self.hidden_dim = hidden_dim
        self.softmax_scale = 1
        self.feat_unfold = True
        self.local_attn = True
        self.ot = False
        self.phi = nn.Parameter(torch.tensor(0.0, requires_grad=True))  # dustbin cost

        # self.grad_nopad = Get_gradient_nopadding()
        self.encoder = models.make(encoder_spec)
        self.projl = nn.Conv2d(self.encoder.out_dim, self.encoder.out_dim, 1, 1, 0)
        self.projr = nn.Conv2d(self.encoder.out_dim, self.encoder.out_dim, 1, 1, 0)
    
        # self.norm_l = LayerNorm2d(self.encoder.out_dim)
        # self.norm_r = LayerNorm2d(self.encoder.out_dim)
        self.coef = nn.Conv2d(2 * self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.freq = nn.Conv2d(2 * self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.conv0 = nn.Conv2d(1 + 3, hidden_dim, 3, padding=1)
        
        self.phase = nn.Linear(2, hidden_dim, bias=False)
        self.imnet = models.make(imnet_spec, args={'in_dim': hidden_dim})
        self.dispnet = models.make(dispnet_spec, args={'in_dim': hidden_dim})
        self.occ_head = nn.Sigmoid()
        self.pe = PositionEncoder(posenc_type='sinusoid',complex_transform=False, enc_dims=hidden_dim, hidden_dims=hidden_dim//2)

    @torch.no_grad()
    def _generate_square_subsequent_mask(self, sz: int):
        """
        Generate a mask which is upper triangular

        :param sz: square matrix size
        :return: diagonal binary mask [sz,sz]
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask[mask == 1] = float('-inf')
        return mask
    
    def M_Relax(self, M, num_pixels):
        _, _, u, v = M.shape
        M = M.view(-1, u, v)
        M_list = []
        M_list.append(M.unsqueeze(1))
        for i in range(num_pixels):
            pad = nn.ZeroPad2d(padding=(0, 0, i+1, 0))
            pad_M = pad(M[:, :-1-i, :])
            M_list.append(pad_M.unsqueeze(1))   # B*H, 1, W, W
        for i in range(num_pixels):
            pad = nn.ZeroPad2d(padding=(0, 0, 0, i+1))
            pad_M = pad(M[:, i+1:, :])
            M_list.append(pad_M.unsqueeze(1))
        M_relaxed = torch.sum(torch.cat(M_list, 1), dim=1)
        return M_relaxed
    
    def _compute_unscaled_pos_shift(self, w: int, device: torch.device):
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

    def _compute_low_res_disp(self, pos_shift: Tensor, attn_weight: Tensor, occ_mask: Tensor):
        """
        Compute low res disparity using the attention weight by finding the most attended pixel and regress within the 3px window

        :param pos_shift: relative pos shift (computed from _compute_unscaled_pos_shift), [1,1,W,W]
        :param attn_weight: attention (computed from _optimal_transport), [N,H,W,W]
        :param occ_mask: ground truth occlusion mask, [N,H,W]
        :return: low res disparity, [N,H,W] and attended similarity sum, [N,H,W]
        """

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
    
    def _sinkhorn(self, attn: Tensor, log_mu: Tensor, log_nu: Tensor, iters: int):
        """
        Sinkhorn Normalization in Log-space as matrix scaling problem.
        Regularization strength is set to 1 to avoid manual checking for numerical issues
        Adapted from SuperGlue (https://github.com/magicleap/SuperGluePretrainedNetwork)

        :param attn: input attention weight, [N,H,W+1,W+1]
        :param log_mu: marginal distribution of left image, [N,H,W+1]
        :param log_nu: marginal distribution of right image, [N,H,W+1]
        :param iters: number of iterations
        :return: updated attention weight
        """

        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for idx in range(iters):
            # scale v first then u to ensure row sum is 1, col sum slightly larger than 1
            v = log_nu - torch.logsumexp(attn + u.unsqueeze(3), dim=2)
            u = log_mu - torch.logsumexp(attn + v.unsqueeze(2), dim=3)

        return attn + u.unsqueeze(3) + v.unsqueeze(2)

    def _optimal_transport(self, attn: Tensor, iters: int):
        """
        Perform Differentiable Optimal Transport in Log-space for stability
        Adapted from SuperGlue (https://github.com/magicleap/SuperGluePretrainedNetwork)

        :param attn: raw attention weight, [N,H,W,W]
        :param iters: number of iterations to run sinkhorn
        :return: updated attention weight, [N,H,W+1,W+1]
        """
        bs, h, w, _ = attn.shape

        # set marginal to be uniform distribution
        marginal = torch.cat([torch.ones([w]), torch.tensor([w]).float()]) / (2 * w)
        log_mu = marginal.log().to(attn.device).expand(bs, h, w + 1)
        log_nu = marginal.log().to(attn.device).expand(bs, h, w + 1)

        # add dustbins
        similarity_matrix = torch.cat([attn, self.phi.expand(bs, h, w, 1).to(attn.device)], -1)
        similarity_matrix = torch.cat([similarity_matrix, self.phi.expand(bs, h, 1, w + 1).to(attn.device)], -2)

        # sinkhorn
        attn_ot = self._sinkhorn(similarity_matrix, log_mu, log_nu, iters)

        # convert back from log space, recover probabilities by normalization 2W
        attn_ot = (attn_ot + torch.log(torch.tensor([2.0 * w]).to(attn.device))).exp()

        return attn_ot

    def _softmax(self, attn: Tensor):
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

    def _compute_low_res_occ(self, matched_attn: Tensor):
        """
        Compute low res occlusion by using inverse of the matched values

        :param matched_attn: updated attention weight without dustbins, [N,H,W,W]
        :return: low res occlusion map, [N,H,W]
        """
        occ_pred = 1.0 - matched_attn
        return occ_pred.squeeze(-1)
    
    def regress_disp(self, attn_weight: Tensor, sampled_cols: Tensor = None, sampled_rows: Tensor = None, \
                     occ_mask: Tensor = None, occ_mask_right: Tensor = None, mode='r2l'):
        """
        Regression head follows steps of
            - compute scale for disparity (if there is downsampling)
            - impose uniqueness constraint by optimal transport
            - compute RR loss
            - regress disparity and occlusion
            - upsample (if there is downsampling) and adjust based on context
        
        :param attn_weight: raw attention weight, [N,H,W,W]
        :param x: input data
        :return: dictionary of predicted values
        """
        # bs, _, h, w = x.shape
        output = {}

        # compute scale
        # scale = 1.0

        # normalize attention to 0-1
        if self.ot:
            # optimal transport
            attn_ot = self._optimal_transport(attn_weight, 10)
        else:
            # softmax
            attn_ot = self._softmax(attn_weight)
        output['attn_ot'] = attn_ot[..., :-1, :-1]
        # compute relative response (RR) at ground truth location
        output['gt_response'] = None

        # compute relative response (RR) at occluded location
        if occ_mask is not None:
            # handle occlusion
            occ_mask = occ_mask
            occ_mask_right = occ_mask_right
            if sampled_cols is not None:
                occ_mask = batched_index_select(occ_mask, 2, sampled_cols)
                occ_mask_right = batched_index_select(occ_mask_right, 2, sampled_cols)
            if sampled_rows is not None:
                occ_mask = batched_index_select(occ_mask, 1, sampled_rows)
                occ_mask_right = batched_index_select(occ_mask_right, 1, sampled_rows)

            output['gt_response_occ_left'] = attn_ot[..., :-1, -1][occ_mask]
            output['gt_response_occ_right'] = attn_ot[..., -1, :-1][occ_mask_right]
        else:
            output['gt_response_occ_left'] = None
            output['gt_response_occ_right'] = None
            occ_mask = occ_mask

        # regress low res disparity
        pos_shift = self._compute_unscaled_pos_shift(attn_weight.shape[2], attn_weight.device)  # NxHxW_leftxW_right
        if mode == 'l2r':
            pos_shift = pos_shift.permute(0, 1, 3, 2)
        disp_pred_low_res, matched_attn = self._compute_low_res_disp(pos_shift, attn_ot[..., :-1, :-1], occ_mask)
        # regress low res occlusion
        occ_pred_low_res = self._compute_low_res_occ(matched_attn)

        # with open('attn_weight.dat', 'wb') as f:
        #     torch.save(attn_ot[0], f)
        # with open('target.dat', 'wb') as f:
        #     torch.save(target, f)

        # upsample and context adjust
        output['disp_pred'] = disp_pred_low_res
        output['occ_pred'] = occ_pred_low_res

        return output

    def gen_feat(self, x_left, x_right, scale):
        self.inp_l, self.inp_r = x_left, x_right
        b,c,h,w = self.inp_l.shape
        x = torch.cat((self.inp_l, self.inp_r), dim=0)
        self.feat_coord = make_coord(self.inp_l.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(self.inp_l.shape[0], 2, *self.inp_l.shape[-2:])
        # self.cost = [
        #     torch.zeros(b, h, w, w).to(x_left.device),
        #     torch.zeros(b, h, w, w).to(x_left.device)
        # ]
        # self.feat_left, self.feat_right, self.disp1, self.disp2, \
        #     (self.M_right_to_left, self.M_left_to_right), (self.V_left, self.V_right) \
        #         = self.encoder(self.inp_l, self.inp_r, self.cost)
        self.feat_left, self.feat_right, self.M_right_to_left, self.M_left_to_right \
            = self.encoder(x)
        
        # #  self.disp_r2l, self.disp_l2r = self.gen_initial_disp(self.M_right_to_left, self.M_left_to_right)
        # attn_mask = self._generate_square_subsequent_mask(w).to(x_left.device)  # generate attn mask
        # masked_M_right_to_left = self.M_right_to_left  + attn_mask[None, None, ...] 
        # masked_M_left_to_right = self.M_left_to_right  + attn_mask[None, None, ...].permute(0, 1, 3, 2)  
        # ######################################## Winner Takes ALL #########################################################
        # out_1 = self.regress_disp(masked_M_right_to_left)
        # out_2 = self.regress_disp(masked_M_left_to_right, mode='l2r')
        # self.disp1, self.occ_left = out_1['disp_pred'].unsqueeze(1), out_2['occ_pred'].unsqueeze(1)
        # self.disp2, self.occ_right = out_1['disp_pred'].unsqueeze(1), out_2['occ_pred'].unsqueeze(1)
        # self.M_right_to_left, self.M_left_to_right = out_1['attn_ot'], out_2['attn_ot']
        # ######################################## disp norm #########################################################
        # # self.disp_l2r, self.occ_left = self.norm_disp(self.disp_l2r, self.occ_left)
        # # self.disp_r2l, self.occ_right = self.norm_disp(self.disp_r2l, self.occ_right)

        # self.disp1 = self.disp1 * scale
        # self.disp2 = self.disp2 * scale

        # # softmax
        self.M_right_to_left = torch.softmax(self.M_right_to_left, dim=-1)
        self.M_left_to_right = torch.softmax(self.M_left_to_right, dim=-1)
        # ######################################## valid mask #########################################################
        M_right_to_left_relaxed = self.M_Relax(self.M_right_to_left, num_pixels=2)
        V_left = torch.bmm(M_right_to_left_relaxed.contiguous().view(-1, w).unsqueeze(1),
                           self.M_left_to_right.permute(0, 1, 3, 2).contiguous().view(-1, w).unsqueeze(2)
                           ).detach().contiguous().view(b, 1, h, w)  # (B*H*Wr) * Wl * 1
        M_left_to_right_relaxed = self.M_Relax(self.M_left_to_right, num_pixels=2)
        V_right = torch.bmm(M_left_to_right_relaxed.contiguous().view(-1, w).unsqueeze(1),  # (B*H*Wl) * 1 * Wr
                            self.M_right_to_left.permute(0, 1, 3, 2).contiguous().view(-1, w).unsqueeze(2)
                                  ).detach().contiguous().view(b, 1, h, w)   # (B*H*Wr) * Wl * 1

        self.occ_left = torch.tanh(5 * V_left) # b,1,h,wl
        self.occ_right = torch.tanh(5 * V_right) # b,1,h,wr

        # ############################################# Sum #################################################################
        index = torch.arange(w).view(1, 1, 1, w).to(self.M_right_to_left.device).float()    # index: 1*1*w*1
        # pos_shift = self._compute_unscaled_pos_shift(self.M_right_to_left.shape[2], self.M_right_to_left.device)  # NxHxW_leftxW_right
        # pos_shift_T = pos_shift.permute(0, 1, 3, 2)
        self.disp1 = torch.sum(self.M_right_to_left * index, dim=-1).view(b, 1, h, w)  # x axis of the corresponding point  b,1,h,w_l
        self.disp2 = torch.sum(self.M_left_to_right * index, dim=-1).view(b, 1, h, w)  # b,1,h,w_r

        self.disp1 = self.disp1 * scale
        self.disp2 = self.disp2 * scale

        return self.feat_left, self.feat_right, self.M_left_to_right, self.M_right_to_left
    
    
    def query_rgb_left(self, coord, cell=None):
        feat_leftW = torch.matmul(self.M_right_to_left, self.projr(self.feat_right).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        feat = torch.cat((self.feat_left, feat_leftW), dim=1)

        eps = 1e-6
        mean_disp_pred = self.disp1.mean()
        std_disp_pred = self.disp1.std() + eps
        disp_pred_normalized = (self.disp1 - mean_disp_pred) / std_disp_pred

        
        # feat_d = torch.cat((self.feat_left, self.disp_l2r), dim=1)

        feat_d = self.conv0(torch.cat((disp_pred_normalized, self.inp_l), dim=1))

        # feat = torch.cat((feat, self.disp_l2r), dim=1)
        # feat = self.feat_left
        # feat = torch.cat((feat, self.disp1), dim=1)
        # key pos
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
        disps = []
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
                q_d = F.grid_sample(
                    feat_d, coord_.flip(-1).unsqueeze(1),
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
                # q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)
                # q_freq = torch.mul(q_freq, rel_coord.unsqueeze(-1))
                # q_freq = torch.sum(q_freq, dim=-2)
                q_freq += self.phase(rel_cell.view((bs * q, -1))).view(bs, q, -1)
                # q_freq = torch.cat((torch.cos(np.pi*q_freq), torch.sin(np.pi*q_freq)), dim=-1)

                inp = torch.mul(q_coef, q_freq)
                inp_d = q_d + q_freq          

                pred = self.imnet(inp.contiguous().view(bs * q, -1)).view(bs, q, -1)
                pred_disp = self.dispnet(inp_d.contiguous().view(bs * q, -1)).view(bs, q, -1)
                # pred_disp = torch.zeros((bs, q, 1)).to(pred.device)
                preds.append(pred)
                disps.append(pred_disp)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t
        
        ret = 0
        ret_disp = 0
        for pred, pred_disp, area in zip(preds, disps, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
            ret_disp = ret_disp + pred_disp * (area / tot_area).unsqueeze(-1)
        rgb = ret + F.grid_sample(self.inp_l, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        disp1_hr = ret_disp + F.grid_sample(disp_pred_normalized, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        mask1_hr = F.grid_sample(self.occ_left, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        disp1_hr = disp1_hr * std_disp_pred + mean_disp_pred
        

        return rgb, disp1_hr, mask1_hr
    
    def query_rgb_right(self, coord, cell=None):
        
        feat_rightW = torch.matmul(self.M_left_to_right, self.projr(self.feat_left).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        feat = torch.cat((self.feat_right, feat_rightW), dim=1)

        
        # feat = self.feat_right + feat_rightW
        # feat_d = torch.cat((self.feat_right, self.disp_r2l), dim=1)
        eps = 1e-6
        mean_disp_pred = self.disp2.mean()
        std_disp_pred = self.disp2.std() + eps
        disp_pred_normalized = (self.disp2 - mean_disp_pred) / std_disp_pred
        

        feat_d = self.conv0(torch.cat((disp_pred_normalized, self.inp_r), dim=1))
        # feat = self.feat_right
        # feat = torch.cat((feat, self.disp_r2l), dim=1)
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
        disps = []
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
                
                q_d = F.grid_sample(
                    feat_d, coord_.flip(-1).unsqueeze(1),
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
                # q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)
                # q_freq = torch.mul(q_freq, rel_coord.unsqueeze(-1))
                # q_freq = torch.sum(q_freq, dim=-2)
                q_freq += self.phase(rel_cell.view((bs * q, -1))).view(bs, q, -1)
                # q_freq = torch.cat((torch.cos(np.pi*q_freq), torch.sin(np.pi*q_freq)), dim=-1)
                
                inp = torch.mul(q_coef, q_freq)
                inp_d = q_d + q_freq             

                pred = self.imnet(inp.contiguous().view(bs * q, -1)).view(bs, q, -1)
                pred_disp = self.dispnet(inp_d.contiguous().view(bs * q, -1)).view(bs, q, -1)
                # pred_disp = torch.zeros((bs, q, 1)).to(pred.device)
                preds.append(pred)
                disps.append(pred_disp)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t
        
        ret = 0
        ret_disp = 0
        for pred, pred_disp, area in zip(preds, disps, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
            ret_disp = ret_disp + pred_disp * (area / tot_area).unsqueeze(-1)

        rgb = ret + F.grid_sample(self.inp_r, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        disp2_hr = ret_disp + F.grid_sample(disp_pred_normalized, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        mask2_hr = F.grid_sample(self.occ_right, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        disp2_hr = disp2_hr * std_disp_pred + mean_disp_pred
        
        return rgb, disp2_hr, mask2_hr
    

    def forward(self, inp_left, inp_right, coord, cell, scale):
        self.gen_feat(inp_left, inp_right, scale)
        return self.query_rgb_left(coord, cell), self.query_rgb_right(coord, cell),\
              (self.M_left_to_right, self.M_right_to_left)