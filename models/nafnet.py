# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

import numpy as np
import math
from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat

def M_Relax(M, num_pixels):
    _, u, v = M.shape
    M_list = []
    M_list.append(M.unsqueeze(1))
    for i in range(num_pixels):
        pad = nn.ZeroPad2d(padding=(0, 0, i+1, 0))
        pad_M = pad(M[:, :-1-i, :])
        M_list.append(pad_M.unsqueeze(1))
    for i in range(num_pixels):
        pad = nn.ZeroPad2d(padding=(0, 0, 0, i+1))
        pad_M = pad(M[:, i+1:, :])
        M_list.append(pad_M.unsqueeze(1))
    M_relaxed = torch.sum(torch.cat(M_list, 1), dim=1)
    return M_relaxed

class CALayer(nn.Module):
    def __init__(self, channel, reduction):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel//reduction, 1, padding=0, bias=True),
                #nn.LeakyReLU(0.1, inplace=True),
                nn.PReLU(),
                nn.Conv2d(channel//reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr= False, compress_ratio=3,squeeze_factor=30):
        super(CAB, self).__init__()
        if is_light_sr: # we use depth-wise conv for light-SR to achieve more efficient
            self.cab = nn.Sequential(
                nn.Conv2d(num_feat, num_feat, 3, 1, 1, groups=num_feat),
                CALayer(num_feat, squeeze_factor)
            )
        else: # for classic SR
            self.cab = nn.Sequential(
                nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
                CALayer(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)

class one_conv(nn.Module):
    def __init__(self, G0, G):
        super(one_conv, self).__init__()
        self.conv = nn.Conv2d(G0, G, kernel_size=3, stride=1, padding=1, bias=True)
        #self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.relu = nn.PReLU()
    def forward(self, x):
        output = self.relu(self.conv(x))
        return torch.cat((x, output), dim=1)


class RDB(nn.Module):
    def __init__(self, G0, C, G):
        super(RDB, self).__init__()
        convs = []
        for i in range(C):
            convs.append(one_conv(G0+i*G, G))
        self.conv = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(G0+C*G, G0, kernel_size=1, stride=1, padding=0, bias=True)
    def forward(self, x):
        out = self.conv(x)
        lff = self.LFF(out)
        return lff + x
    
class RDG(nn.Module):
    def __init__(self, G0, C, G, n_RDB):
        super(RDG, self).__init__()
        self.n_RDB = n_RDB
        RDBs = []
        for i in range(n_RDB):
            RDBs.append(RDB(G0, C, G))
        self.RDB = nn.Sequential(*RDBs)
        self.conv = nn.Conv2d(G0*n_RDB, G0, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        buffer = x
        temp = []
        for i in range(self.n_RDB):
            buffer = self.RDB[i](buffer)
            temp.append(buffer)
        buffer_cat = torch.cat(temp, dim=1)
        out = self.conv(buffer_cat)
        return out

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

# handle multiple input
class MySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs 

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
    

class SCAM(nn.Module):
    '''
    Stereo Cross Attention Module (SCAM)
    '''
    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5

        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, x_l, x_r, cost):
        b, c, h, w = x_l.shape
        Q_l = self.l_proj1(self.norm_l(x_l)).permute(0, 2, 3, 1)  # B, H, Wl, c
        Q_r_T = self.r_proj1(self.norm_r(x_r)).permute(0, 2, 1, 3) # B, H, c, Wr (transposed)

        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)  # B, H, Wl, c
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)  # B, H, Wr, c

        # (B, H, Wl, c) x (B, H, c, Wr) -> (B, H, Wl, Wr)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  #B, H, Wl, c
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l) #B, H, Wr, c
        
        cost[0] += attention.contiguous()
        cost[1] += attention.contiguous().permute(0, 1, 3, 2)   
        # scale
        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma
        return x_l + F_r2l, x_r + F_l2r, cost

class DropModule(nn.Module):
    def __init__(self, drop_rate, module):
        super().__init__()
        self.drop_rate = drop_rate
        self.module = module

    def forward(self, *feats):
        if self.training and np.random.rand() < self.drop_rate:
            return feats

        new_feats = self.module(*feats)
        factor = 1. / (1 - self.drop_rate) if self.training else 1.

        if self.training and factor != 1.:
            new_feats = tuple([x+factor*(new_x-x) for x, new_x in zip(feats, new_feats)])
        return new_feats
    

class NAFBlockSR(nn.Module):
    '''
    NAFBlock for Super-Resolution
    '''
    def __init__(self, c, fusion=False, drop_out_rate=0.):
        super().__init__()
        self.blk = NAFBlock(c, drop_out_rate=drop_out_rate).cuda()
        self.fusion = SCAM(c).cuda() if fusion else None

    def forward(self, x_left, x_right, cost):
        feat_left = self.blk(x_left)
        feat_right = self.blk(x_right)

        if self.fusion:
            feat_left, feat_right, cost = self.fusion(feat_left, feat_right, cost)

        # feats = tuple([self.blk(x) for x in feats])
        # if self.fusion:
        #     feats = self.fusion(*feats)
        return feat_left, feat_right, cost

class NAFNetSR(nn.Module):
    '''
    NAFNet for Super-Resolution
    '''
    def __init__(self, args):
        super().__init__()
        self.dual = args.dual    # dual input for stereo SR (left view, right view)
        self.width = args.width
        self.drop_out_rate = args.drop_out_rate
        self.no_upsampling = args.no_upsampling
        self.body = nn.ModuleList()
        self.num_blks = args.num_blks
        self.intro = nn.Conv2d(in_channels=args.img_channel, out_channels=args.width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        # self.transition = RDG(G0=args.width, C=4, G=24, n_RDB=4)
        for i in range(self.num_blks):
            self.body.append(
                DropModule(
                    self.drop_out_rate, 
                    NAFBlockSR(
                        self.width, 
                        fusion=(args.fusion_from <= i and i <= args.fusion_to), 
                        drop_out_rate=self.drop_out_rate
                ))
            )
        self.softmax = nn.Softmax(-1)
        # self.CALayer = CALayer(self.width, self.width//8)
        # self.resblock = RDB(G0=self.width, C=6, G=24)
        if args.no_upsampling:
            self.out_dim = self.width
        else:
            self.out_dim = 3

    def forward(self, x_left, x_right, cost):
        x_size = (x_left.shape[2], x_left.shape[3])
        feat_left = self.intro(x_left)
        feat_right = self.intro(x_right)
        shallow_feat_l, shallow_feat_r = feat_left, feat_right
        b, c, h, w = feat_left.shape

        for i in range(len(self.body)):
            feat_left, feat_right, cost = self.body[i](feat_left, feat_right, cost)
        
        feat_left = feat_left + shallow_feat_l
        feat_right = feat_right + shallow_feat_r

        M_right_to_left = self.softmax(cost[0])                                  # (B*H) * Wl * Wr
        M_left_to_right = self.softmax(cost[1])                                  # (B*H) * Wr * Wl

        # M_right_to_left_relaxed = M_Relax(M_right_to_left, num_pixels=2)
        # V_left = torch.bmm(M_right_to_left_relaxed.contiguous().view(-1, w).unsqueeze(1),
        #                    M_left_to_right.permute(0, 2, 1).contiguous().view(-1, w).unsqueeze(2)
        #                    ).detach().contiguous().view(b, 1, h, w)  # (B*H*Wr) * Wl * 1
        # M_left_to_right_relaxed = M_Relax(M_left_to_right, num_pixels=2)
        # V_right = torch.bmm(M_left_to_right_relaxed.contiguous().view(-1, w).unsqueeze(1),  # (B*H*Wl) * 1 * Wr
        #                     M_right_to_left.permute(0, 2, 1).contiguous().view(-1, w).unsqueeze(2)
        #                           ).detach().contiguous().view(b, 1, h, w)   # (B*H*Wr) * Wl * 1

        # V_left_tanh = torch.tanh(5 * V_left)
        # V_right_tanh = torch.tanh(5 * V_right)

        # x_leftT = torch.bmm(M_right_to_left, feat_left.permute(0, 2, 3, 1).contiguous().view(-1, w, c)
        #                     ).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)                           #  B, C0, H0, W0
        # x_rightT = torch.bmm(M_left_to_right, feat_right.permute(0, 2, 3, 1).contiguous().view(-1, w, c)
        #                     ).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)                              #  B, C0, H0, W0
        # out_left = feat_left * (1 - V_left_tanh.repeat(1, c, 1, 1)) + x_leftT * V_left_tanh.repeat(1, c, 1, 1)
        # out_right = feat_left * (1 - V_right_tanh.repeat(1, c, 1, 1)) +  x_rightT * V_right_tanh.repeat(1, c, 1, 1)

        # index = torch.arange(w).view(1, 1, 1, w).to(M_right_to_left.device).float()    # index: 1*1*1*w
        # disp1 = torch.sum(M_right_to_left * index, dim=-1).view(b, 1, h, w) # x axis of the corresponding point
        # disp2 = torch.sum(M_left_to_right * index, dim=-1).view(b, 1, h, w)

        # out_left = self.CALayer(feat_left + self.resblock(out_left - feat_left))
        # out_right = self.CALayer(feat_right + self.resblock(out_right - feat_right))
        
        return feat_left, feat_right, M_left_to_right, M_right_to_left





@register('nafnet')
def make_nafnet(up_scale=4, width=48, num_blks=16, img_channel=3, drop_path_rate=0., drop_out_rate=0., fusion_from=-1, fusion_to=16, dual=True, no_upsampling =True):
    args = Namespace()
    args.up_scale = up_scale
    args.width = width
    args.num_blks = num_blks

    args.img_channel = img_channel
    args.drop_path_rate = drop_path_rate

    args.drop_out_rate = drop_out_rate
    args.fusion_from = fusion_from
    args.fusion_to = fusion_to
    args.dual = dual
    args.no_upsampling = no_upsampling

    return NAFNetSR(args)
