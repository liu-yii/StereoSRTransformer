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

class ResBlock(nn.Module):
    def __init__(self, n_feats: int, expansion_ratio: int, res_scale: int = 1.0):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.module = nn.Sequential(
            weight_norm(nn.Conv2d(n_feats + 1, n_feats * expansion_ratio, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            weight_norm(nn.Conv2d(n_feats * expansion_ratio, n_feats, kernel_size=3, padding=1))
        )

    def forward(self, x: torch.Tensor, disp: torch.Tensor):
        return x + self.module(torch.cat([disp, x], dim=1)) * self.res_scale

class ContextAdjustmentLayer(nn.Module):
    """
    Adjust the disp and occ based on image context, design loosely follows https://github.com/JiahuiYu/wdsr_ntire2018
    """

    def __init__(self, num_blocks=8, feature_dim=16, expansion=3):
        super().__init__()
        self.num_blocks = num_blocks

        # disp head
        self.in_conv = nn.Conv2d(4, feature_dim, kernel_size=3, padding=1)
        self.layers = nn.ModuleList([ResBlock(feature_dim, expansion) for _ in range(num_blocks)])
        self.out_conv = nn.Conv2d(feature_dim, 1, kernel_size=3, padding=1)

        # occ head
        self.occ_head = nn.Sequential(
            weight_norm(nn.Conv2d(1 + 3, feature_dim, kernel_size=3, padding=1)),
            weight_norm(nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            weight_norm(nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)),
            weight_norm(nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, disp_raw: Tensor, occ_raw: Tensor, img: Tensor):
        """
        :param disp_raw: raw disparity, [N,1,H,W]
        :param occ_raw: raw occlusion mask, [N,1,H,W]
        :param img: input left image, [N,3,H,W]
        :return:
            disp_final: final disparity [N,1,H,W]
            occ_final: final occlusion [N,1,H,W] 
        """""
        feat = self.in_conv(torch.cat([disp_raw, img], dim=1))
        for layer in self.layers:
            feat = layer(feat, disp_raw)
        disp_res = self.out_conv(feat)
        disp_final = disp_raw + disp_res

        occ_final = self.occ_head(torch.cat([occ_raw, img], dim=1))

        return disp_final, occ_final

class RegressionHead(nn.Module):
    """
    Regress Disparity and Occ mask
    """
    def __init__(self, cal: nn.Module, ot: bool = True):
        super(RegressionHead, self).__init__()
        self.cal = cal
        self.ot = ot
        self.phi = nn.Parameter(torch.tensor(0.0, requires_grad=True))  # dustbin cost

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

    def _compute_gt_location(self, scale: int, sampled_cols: Tensor, sampled_rows: Tensor,
                             attn_weight: Tensor, disp: Tensor):
        """
        Find target locations using ground truth disparity.
        Find ground truth response at those locations using attention weight.

        :param scale: high-res to low-res disparity scale
        :param sampled_cols: index to downsample columns
        :param sampled_rows: index to downsample rows
        :param attn_weight: attention weight (output from _optimal_transport), [N,H,W,W]
        :param disp: ground truth disparity
        :return: response at ground truth location [N,H,W,1] and target ground truth locations [N,H,W,1]
        """
        # compute target location at full res
        _, _, w = disp.size()
        pos_l = torch.linspace(0, w - 1, w)[None,].to(disp.device)  # 1 x 1 x W (left)
        target = (pos_l - disp)[..., None]  # N x H x W (left) x 1

        if sampled_cols is not None:
            target = batched_index_select(target, 2, sampled_cols)
        if sampled_rows is not None:
            target = batched_index_select(target, 1, sampled_rows)
        target = target / scale  # scale target location

        # compute ground truth response location for rr loss
        gt_response = torch_1d_sample(attn_weight, target, 'linear')  # NxHxW_left

        return gt_response, target

    def _upsample(self, x: NestedTensor, disp_pred: Tensor, occ_pred: Tensor, scale: int):
        """
        Upsample the raw prediction to full resolution

        :param x: input data
        :param disp_pred: predicted disp at low res
        :param occ_pred: predicted occlusion at low res
        :param scale: high-res to low-res disparity scale
        :return: high res disp and occ prediction
        """
        _, _, h, w = x.left.size()

        # scale disparity
        disp_pred_attn = disp_pred * scale

        # upsample
        disp_pred = F.interpolate(disp_pred_attn[None,], size=(h, w), mode='nearest')  # N x 1 x H x W
        occ_pred = F.interpolate(occ_pred[None,], size=(h, w), mode='nearest')  # N x 1 x H x W

        if self.cal is not None:
            # normalize disparity
            eps = 1e-6
            mean_disp_pred = disp_pred.mean()
            std_disp_pred = disp_pred.std() + eps
            disp_pred_normalized = (disp_pred - mean_disp_pred) / std_disp_pred

            # normalize occlusion mask
            occ_pred_normalized = (occ_pred - 0.5) / 0.5

            disp_pred_normalized, occ_pred = self.cal(disp_pred_normalized, occ_pred_normalized, x.left)  # N x H x W

            disp_pred_final = disp_pred_normalized * std_disp_pred + mean_disp_pred
        else:
            disp_pred_final = disp_pred.squeeze(1)
            disp_pred_attn = disp_pred_attn.squeeze(1)

        return disp_pred_final.squeeze(1), disp_pred_attn.squeeze(1), occ_pred.squeeze(1)

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

    def forward(self, attn_weight: Tensor, x: NestedTensor):
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
        bs, _, h, w = x.left.size()
        output = {}

        # compute scale
        if x.sampled_cols is not None:
            scale = x.left.size(-1) / float(x.sampled_cols.size(-1))
        else:
            scale = 1.0

        # normalize attention to 0-1
        if self.ot:
            # optimal transport
            attn_ot = self._optimal_transport(attn_weight, 10)
        else:
            # softmax
            attn_ot = self._softmax(attn_weight)

        # compute relative response (RR) at ground truth location
        if x.disp is not None:
            # find ground truth response (gt_response) and location (target)
            output['gt_response'], target = self._compute_gt_location(scale, x.sampled_cols, x.sampled_rows,
                                                                      attn_ot[..., :-1, :-1], x.disp)
        else:
            output['gt_response'] = None

        # compute relative response (RR) at occluded location
        if x.occ_mask is not None:
            # handle occlusion
            occ_mask = x.occ_mask
            occ_mask_right = x.occ_mask_right
            if x.sampled_cols is not None:
                occ_mask = batched_index_select(occ_mask, 2, x.sampled_cols)
                occ_mask_right = batched_index_select(occ_mask_right, 2, x.sampled_cols)
            if x.sampled_rows is not None:
                occ_mask = batched_index_select(occ_mask, 1, x.sampled_rows)
                occ_mask_right = batched_index_select(occ_mask_right, 1, x.sampled_rows)

            output['gt_response_occ_left'] = attn_ot[..., :-1, -1][occ_mask]
            output['gt_response_occ_right'] = attn_ot[..., -1, :-1][occ_mask_right]
        else:
            output['gt_response_occ_left'] = None
            output['gt_response_occ_right'] = None
            occ_mask = x.occ_mask

        # regress low res disparity
        pos_shift = self._compute_unscaled_pos_shift(attn_weight.shape[2], attn_weight.device)  # NxHxW_leftxW_right
        disp_pred_low_res, matched_attn = self._compute_low_res_disp(pos_shift, attn_ot[..., :-1, :-1], occ_mask)
        # regress low res occlusion
        occ_pred_low_res = self._compute_low_res_occ(matched_attn)

        # with open('attn_weight.dat', 'wb') as f:
        #     torch.save(attn_ot[0], f)
        # with open('target.dat', 'wb') as f:
        #     torch.save(target, f)

        # upsample and context adjust
        if x.sampled_cols is not None:
            output['disp_pred'], output['disp_pred_low_res'], output['occ_pred'] = self._upsample(x, disp_pred_low_res,
                                                                                                  occ_pred_low_res,
                                                                                                  scale)
        else:
            output['disp_pred'] = disp_pred_low_res
            output['occ_pred'] = occ_pred_low_res

        return output
# def get_embed_fns(max_freq):
#     """
#     N,bsize,1 ---> N,bsize,2n+1
#     """
#     embed_fns = []
#     embed_fns.append(lambda x: torch.ones((x.shape[0], x.shape[1], 1)))  # x: N,bsize,1
#     for i in range(1, max_freq + 1):
#         embed_fns.append(lambda x, freq=i: math.sqrt(2) * torch.cos(x[:, :, 0] * freq).unsqueeze(-1))
#         embed_fns.append(lambda x, freq=i: math.sqrt(2) * torch.sin(x[:, :, 0] * freq).unsqueeze(-1))
#     return embed_fns

# class OPE(nn.Module):
#     def __init__(self, max_freq, omega):
#         super(OPE, self).__init__()
#         self.max_freq = max_freq
#         self.omega = omega
#         self.embed_fns = get_embed_fns(self.max_freq)

#     def embed(self, inputs):
#         """
#         N,bsize,1 ---> N,bsize,1,2n+1
#         """
#         res = torch.cat([fn(inputs * self.omega).to(inputs.device) for fn in self.embed_fns], -1)
#         return res.unsqueeze(-2)

#     def forward(self, coords):
#         """
#         N,bsize,2 ---> N,bsize,(2n+1)^2
#         """
#         x_coord = coords[:, :, 0].unsqueeze(-1)
#         y_coord = coords[:, :, 1].unsqueeze(-1)
#         X = self.embed(x_coord)
#         Y = self.embed(y_coord)
#         ope_mat = torch.matmul(X.transpose(2, 3), Y)
#         ope_flat = ope_mat.view(ope_mat.shape[0], ope_mat.shape[1], -1)
#         return ope_flat
    


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
        self.ot = True
        self.phi = nn.Parameter(torch.tensor(0.0, requires_grad=True))  # dustbin cost

        # self.grad_nopad = Get_gradient_nopadding()
        self.encoder = models.make(encoder_spec)
        self.projl = nn.Conv2d(self.encoder.out_dim, self.encoder.out_dim, 1, 1, 0)
        self.projr = nn.Conv2d(self.encoder.out_dim, self.encoder.out_dim, 1, 1, 0)
    
        # self.norm_l = LayerNorm2d(self.encoder.out_dim)
        # self.norm_r = LayerNorm2d(self.encoder.out_dim)
        self.coef = nn.Conv2d(2 * self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.freq = nn.Conv2d(2 * self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.conv0 = nn.Conv2d(2, hidden_dim, 3, padding=1)
        
        self.phase = nn.Linear(2, hidden_dim, bias=False)
        self.imnet = models.make(imnet_spec, args={'in_dim': hidden_dim})
        self.dispnet = models.make(dispnet_spec, args={'in_dim': hidden_dim})
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
    
    def _compute_unscaled_pos_shift(self, w: int, device: torch.device):
        """
        Compute relative difference between each pixel location from left image to right image, to be used to calculate
        disparity

        :param w: image width
        :param device: torch device
        :return: relative pos shifts
        """
        pos_r = torch.linspace(0, 2, w)[None, None, None, :].to(device)  # 1 x 1 x 1 x W_right
        pos_l = torch.linspace(0, 2, w)[None, None, :, None].to(device)  # 1 x 1 x W_left x1
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
    
    def regress_disp(self, attn_weight: Tensor, x: Tensor, sampled_cols: Tensor = None, sampled_rows: Tensor = None, \
                     occ_mask: Tensor = None, occ_mask_right: Tensor = None, mode='l2r'):
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
        bs, _, h, w = x.shape
        output = {}

        # compute scale
        scale = 1.0

        # normalize attention to 0-1
        if self.ot:
            # optimal transport
            attn_ot = self._optimal_transport(attn_weight, 10)
        else:
            # softmax
            attn_ot = self._softmax(attn_weight)

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
        if mode == 'r2l':
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

    def gen_feat(self, x_left, x_right):
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
        self.feat_left, self.feat_right, self.M_left_to_right, self.M_right_to_left \
            = self.encoder(x)
        # self.M_right_to_left = torch.softmax(self.M_right_to_left, dim=-1)    # b,h,w_l,w_r
        # self.M_left_to_right = torch.softmax(self.M_left_to_right, dim=-1)    # b,h,w_r,w_l
        
        #  self.disp_r2l, self.disp_l2r = self.gen_initial_disp(self.M_right_to_left, self.M_left_to_right)
        attn_mask = self._generate_square_subsequent_mask(w).to(x_left.device)  # generate attn mask
        masked_M_right_to_left = self.M_right_to_left + attn_mask[None, None, ...] 
        masked_M_left_to_right = self.M_left_to_right + attn_mask[None, None, ...].permute(0,1,3,2)  
        ######################################## Winner Takes ALL #########################################################
        out_l2r = self.regress_disp(masked_M_right_to_left, self.inp_l)
        out_r2l = self.regress_disp(masked_M_left_to_right, self.inp_r, mode='r2l')
        self.disp_l2r, self.occ_left = out_l2r['disp_pred'].unsqueeze(1), out_l2r['occ_pred'].unsqueeze(1)
        self.disp_r2l, self.occ_right = out_r2l['disp_pred'].unsqueeze(1), out_r2l['occ_pred'].unsqueeze(1)
        # ######################################## valid mask #########################################################
        # M_right_to_left_relaxed = self.M_Relax(self.M_right_to_left, num_pixels=2)
        # V_left = torch.bmm(M_right_to_left_relaxed.contiguous().view(-1, w).unsqueeze(1),
        #                    self.M_left_to_right.permute(0, 1, 3, 2).contiguous().view(-1, w).unsqueeze(2)
        #                    ).detach().contiguous().view(b, 1, h, w)  # (B*H*Wr) * Wl * 1
        # M_left_to_right_relaxed = self.M_Relax(self.M_left_to_right, num_pixels=2)
        # V_right = torch.bmm(M_left_to_right_relaxed.contiguous().view(-1, w).unsqueeze(1),  # (B*H*Wl) * 1 * Wr
        #                     self.M_right_to_left.permute(0, 1, 3, 2).contiguous().view(-1, w).unsqueeze(2)
        #                           ).detach().contiguous().view(b, 1, h, w)   # (B*H*Wr) * Wl * 1

        # self.V_left_tanh = torch.tanh(5 * V_left) # b,1,h,wl
        # self.V_right_tanh = torch.tanh(5 * V_right) # b,1,h,wr

        # ############################################# Sum #################################################################
        # index = torch.linspace(0, 2, w).view(1, 1, 1, w).to(self.M_right_to_left.device).float()    # index: 1*1*w*1
        # self.disp_l2r = torch.sum(self.M_right_to_left * index, dim=-1).view(b, 1, h, w) # x axis of the corresponding point  b,1,h,w_l
        # self.disp_r2l = torch.sum(self.M_left_to_right * index, dim=-1).view(b, 1, h, w) # b,1,h,w_r

        return self.feat_left, self.feat_right, self.M_left_to_right, self.M_right_to_left
    
    
    def query_rgb_left(self, coord, cell=None):
        feat_leftW = torch.matmul(self.M_right_to_left, self.projr(self.feat_right).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        feat = torch.cat((self.feat_left, feat_leftW), dim=1)

        # feat_d = torch.cat((self.feat_left, self.disp_l2r), dim=1)
        feat_d = self.conv0(torch.cat((self.disp_l2r, self.occ_left), dim=1))

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
                inp_d = torch.mul(q_d, q_freq)           

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
        disp_l2r_h = ret_disp[:,:,0].unsqueeze(-1) + F.grid_sample(self.disp_l2r, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        mask_l2r_h = ret_disp[:,:,1].unsqueeze(-1) + F.grid_sample(self.occ_left, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        return rgb, disp_l2r_h, mask_l2r_h
    
    def query_rgb_right(self, coord, cell=None):
        
        feat_rightW = torch.matmul(self.M_left_to_right, self.projr(self.feat_left).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        feat = torch.cat((self.feat_right, feat_rightW), dim=1)

        
        # feat = self.feat_right + feat_rightW
        # feat_d = torch.cat((self.feat_right, self.disp_r2l), dim=1)
        feat_d = self.conv0(torch.cat((self.disp_r2l, self.occ_right), dim=1))
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
                inp_d = torch.mul(q_d, q_freq)             

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
        disp_r2l_h = ret_disp[:,:,0].unsqueeze(-1) + F.grid_sample(self.disp_r2l, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        mask_r2l_h = ret_disp[:,:,1].unsqueeze(-1) + F.grid_sample(self.occ_right, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        return rgb, disp_r2l_h, mask_r2l_h
    

    def forward(self, inp_left, inp_right, coord, cell):
        self.gen_feat(inp_left, inp_right)
        return self.query_rgb_left(coord, cell), self.query_rgb_right(coord, cell),\
              (self.M_left_to_right, self.M_right_to_left)