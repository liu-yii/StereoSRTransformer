import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord
from .crossattention_arch import CrossScaleAttention
# from .arch_util import PositionalEncoding
from .positionencoder import PositionEncoder
from .nafnet import CALayer, RDB, LayerNorm2d
from skimage import morphology

import numpy as np
import math

# Morphological Operations
def morphologic_process(mask):
    b, _, _, _ = mask.shape
    mask = ~mask
    mask_np = mask.cpu().numpy().astype(bool)
    mask_np = morphology.remove_small_objects(mask_np, 20, 2)
    mask_np = morphology.remove_small_holes(mask_np, 10, 2)
    for idx in range(b):
        mask_np[idx, 0, :, :] = morphology.binary_closing(mask_np[idx, 0, :, :], morphology.disk(3))
    mask_np = 1-mask_np
    mask_np = mask_np.astype(float)

    return torch.from_numpy(mask_np).float().to(mask.device)


# Disparity Regression
def regress_disp(att, valid_mask):
    '''
    :param att:         B * H * W * W
    :param valid_mask:  B * 1 * H * W
    '''
    b, h, w, _ = att.shape
    index = torch.arange(w).view(1, 1, 1, w).to(att.device).float()    # index: 1*1*1*w
    disp_ini = index - torch.sum(att * index, dim=-1).view(b, 1, h, w)

    # partial conv
    filter1 = torch.zeros(1, 3).to(att.device)
    filter1[0, 0] = 1
    filter1[0, 1] = 1
    filter1 = filter1.view(1, 1, 1, 3)

    filter2 = torch.zeros(1, 3).to(att.device)
    filter2[0, 1] = 1
    filter2[0, 2] = 1
    filter2 = filter2.view(1, 1, 1, 3)

    valid_mask_0 = valid_mask
    disp = disp_ini * valid_mask_0

    valid_mask_num = 1
    while valid_mask_num > 0:
        valid_mask_1 = F.conv2d(valid_mask_0, filter1, padding=[0, 1])
        disp = disp * valid_mask_0 + \
               F.conv2d(disp, filter1, padding=[0, 1]) / (valid_mask_1 + 1e-4) * ((valid_mask_1 > 0).float() - valid_mask_0)
        valid_mask_num = (valid_mask_1 > 0).float().sum() - valid_mask_0.sum()
        valid_mask_0 = (valid_mask_1 > 0).float()

    valid_mask_num = 1
    while valid_mask_num > 0:
        valid_mask_1 = F.conv2d(valid_mask_0, filter2, padding=[0, 1])
        disp = disp * valid_mask_0 + \
               F.conv2d(disp, filter2, padding=[0, 1]) / (valid_mask_1 + 1e-4) * ((valid_mask_1 > 0).float() - valid_mask_0)
        valid_mask_num = (valid_mask_1 > 0).float().sum() - valid_mask_0.sum()
        valid_mask_0 = (valid_mask_1 > 0).float()

    return disp_ini * valid_mask + disp * (1 - valid_mask)

def get_embed_fns(max_freq):
    """
    N,bsize,1 ---> N,bsize,2n+1
    """
    embed_fns = []
    embed_fns.append(lambda x: torch.ones((x.shape[0], x.shape[1], 1)))  # x: N,bsize,1
    for i in range(1, max_freq + 1):
        embed_fns.append(lambda x, freq=i: math.sqrt(2) * torch.cos(x[:, :, 0] * freq).unsqueeze(-1))
        embed_fns.append(lambda x, freq=i: math.sqrt(2) * torch.sin(x[:, :, 0] * freq).unsqueeze(-1))
    return embed_fns

class OPE(nn.Module):
    def __init__(self, max_freq, omega):
        super(OPE, self).__init__()
        self.max_freq = max_freq
        self.omega = omega
        self.embed_fns = get_embed_fns(self.max_freq)

    def embed(self, inputs):
        """
        N,bsize,1 ---> N,bsize,1,2n+1
        """
        res = torch.cat([fn(inputs * self.omega).to(inputs.device) for fn in self.embed_fns], -1)
        return res.unsqueeze(-2)

    def forward(self, coords):
        """
        N,bsize,2 ---> N,bsize,(2n+1)^2
        """
        x_coord = coords[:, :, 0].unsqueeze(-1)
        y_coord = coords[:, :, 1].unsqueeze(-1)
        X = self.embed(x_coord)
        Y = self.embed(y_coord)
        ope_mat = torch.matmul(X.transpose(2, 3), Y)
        ope_flat = ope_mat.view(ope_mat.shape[0], ope_mat.shape[1], -1)
        return ope_flat
    


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
        # self.grad_nopad = Get_gradient_nopadding()
        self.encoder = models.make(encoder_spec)
        self.projl = nn.Conv2d(self.encoder.out_dim, self.encoder.out_dim, 1, 1, 0)
        self.projr = nn.Conv2d(self.encoder.out_dim, self.encoder.out_dim, 1, 1, 0)
        # self.norm_l = LayerNorm2d(self.encoder.out_dim)
        # self.norm_r = LayerNorm2d(self.encoder.out_dim)
        self.coef = nn.Conv2d(2 * self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.freq = nn.Conv2d(2 * self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.conv0 = nn.Conv2d(1, hidden_dim, 1, padding=1)
        
        self.phase = nn.Linear(2, hidden_dim, bias=False)
        self.imnet = models.make(imnet_spec, args={'in_dim': hidden_dim})
        self.dispnet = models.make(dispnet_spec, args={'in_dim': hidden_dim})
        self.pe = PositionEncoder(posenc_type='sinusoid',complex_transform=False, enc_dims=hidden_dim, hidden_dims=hidden_dim//2)


    def gen_initial_disp(self, cost1, cost2, max_disp = 0):
        cost_right2left, cost_left2right = cost2, cost1
        b, h, w, _ = cost_right2left.shape

        # M_right2left
        ## exclude negative disparities & disparities larger than max_disp (if available)
        cost_right2left = torch.tril(cost_right2left)
        if max_disp > 0:
            cost_right2left = cost_right2left - torch.tril(cost_right2left, -max_disp)
        cost_right2left = torch.exp(cost_right2left - cost_right2left.max(-1)[0].unsqueeze(-1))
        cost_right2left = torch.tril(cost_right2left)
        if max_disp > 0:
            cost_right2left = cost_right2left - torch.tril(cost_right2left, -max_disp)
        att_right2left = cost_right2left / (cost_right2left.sum(-1, keepdim=True) + 1e-8)

        # M_left2right
        ## exclude negative disparities & disparities larger than max_disp (if available)
        cost_left2right = torch.triu(cost_left2right)
        if max_disp > 0:
            cost_left2right = cost_left2right - torch.triu(cost_left2right, max_disp)
        cost_left2right = torch.exp(cost_left2right - cost_left2right.max(-1)[0].unsqueeze(-1))
        cost_left2right = torch.triu(cost_left2right)
        if max_disp > 0:
            cost_left2right = cost_left2right - torch.triu(cost_left2right, max_disp)
        att_left2right = cost_left2right / (cost_left2right.sum(-1, keepdim=True) + 1e-8)

        # valid mask (left image)
        valid_mask_left = torch.sum(att_left2right.detach(), -2) > 0.1
        valid_mask_left = valid_mask_left.view(b, 1, h, w)
        valid_mask_left = morphologic_process(valid_mask_left)

        # valid mask (right image)
        valid_mask_right = torch.sum(att_right2left.detach(), -2) > 0.1
        valid_mask_right = valid_mask_right.view(b, 1, h, w)
        valid_mask_right = morphologic_process(valid_mask_right)

        # disparity
        disp_r2l = regress_disp(att_right2left, valid_mask_left)
        disp_l2r = regress_disp(att_left2right, valid_mask_right)
        return disp_r2l, disp_l2r


    def gen_pos_shift(self, w, device):
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
            
    def gen_raw_disp(self, attn_weight, pos_shift, occ_mask = None):
        # b, c, h, w = size
        # find high response area
        high_response = torch.argmax(attn_weight, dim=-1)  # NxHxW

        # build 3 px local window
        response_range = torch.stack([high_response - 1, high_response, high_response + 1], dim=-1)  # NxHxWx3

        # attention with re-weighting
        attn_weight_pad = F.pad(attn_weight, [1, 1], value=0.0)  # N x H x W_left x (W_right+2)
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

    def gen_feat(self, x_left, x_right):
        self.inp_l, self.inp_r = x_left, x_right
        b,c,h,w = self.inp_l.shape
        x = torch.cat((self.inp_l, self.inp_r), dim=0)
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
            = self.encoder(x, self.cost)
        # self.M_right_to_left = torch.softmax(self.M_right_to_left, dim=-1)    # b,h,w_l,w_r
        # self.M_left_to_right = torch.softmax(self.M_left_to_right, dim=-1)    # b,h,w_r,w_l
        
        #  self.disp_r2l, self.disp_l2r = self.gen_initial_disp(self.M_right_to_left, self.M_left_to_right)

        ######################################## Winner Takes ALL #########################################################
       
        pos_shift = self.gen_pos_shift(w, x_left.device)
        self.disp_l2r, _ = self.gen_raw_disp(self.M_right_to_left, pos_shift)
        self.disp_r2l, _ = self.gen_raw_disp(self.M_left_to_right, pos_shift.permute(0, 1, 3, 2))
        self.disp_r2l = self.disp_r2l.view(b, 1, h, w)
        self.disp_l2r = self.disp_l2r.view(b, 1, h, w)

        ############################################# Sum #################################################################
        # index = torch.linspace(0, 2, w).view(1, 1, 1, w).to(self.M_right_to_left.device).float()    # index: 1*1*w*1
        # self.disp_l2r = torch.sum(self.M_right_to_left * index, dim=-1).view(b, 1, h, w) # x axis of the corresponding point  b,1,h,w_l
        # self.disp_r2l = torch.sum(self.M_left_to_right * index, dim=-1).view(b, 1, h, w) # b,1,h,w_r

        # normalize, relative disparity
        self.disp_l2r = self.disp_l2r
        self.disp_r2l = self.disp_r2l

        return self.feat_left, self.feat_right, self.M_left_to_right, self.M_right_to_left
    
    #     self.disp1 = self.disp1 - torch.mean(self.disp1, dim=(2, 3), keepdim=True)
    #     self.disp2 = self.disp2 - torch.mean(self.disp2, dim=(2, 3), keepdim=True)

    #     return self.disp_l2r, self.disp_r2l
    
    def query_rgb_left(self, coord, cell=None):
        feat_leftW = torch.matmul(self.M_right_to_left, self.projr(self.feat_right).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        feat = torch.cat((self.feat_left, feat_leftW), dim=1)

        # feat_d = torch.cat((self.feat_left, self.disp_l2r), dim=1)
        feat_d = self.conv0(self.disp_l2r)

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
        disp_l2r_h = ret_disp + F.grid_sample(self.disp_l2r, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        
        return rgb, disp_l2r_h
    
    def query_rgb_right(self, coord, cell=None):
        
        feat_rightW = torch.matmul(self.M_left_to_right, self.projr(self.feat_left).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        feat = torch.cat((self.feat_right, feat_rightW), dim=1)

        
        # feat = self.feat_right + feat_rightW
        # feat_d = torch.cat((self.feat_right, self.disp_r2l), dim=1)
        feat_d = self.conv0(self.disp_r2l)
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
        disp_r2l_h = ret_disp + F.grid_sample(self.disp_r2l, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                      padding_mode='border', align_corners=False)[:, :, 0, :] \
                      .permute(0, 2, 1)
        return rgb, disp_r2l_h
    

    def forward(self, inp_left, inp_right, coord, cell):
        self.gen_feat(inp_left, inp_right)
        return self.query_rgb_left(coord, cell), self.query_rgb_right(coord, cell)