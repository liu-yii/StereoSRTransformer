# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

import numpy as np
from argparse import Namespace
from models.crossarch import MultiheadAttentionRelative
from models.positionencoder import PositionEncodingSine1DRelative
import torch
import torch.nn as nn

from models import register

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


class CrossAttention(nn.Module):
    """
    Cross attention layer
    """
    def __init__(self, hidden_dim: int, nhead: int):
        super().__init__()
        self.cross_attn = MultiheadAttentionRelative(hidden_dim, nhead)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    def forward(self, feat_left, feat_right,
                pos = None,
                pos_indexes = None,
                last_layer = False):
        """
        :param feat_left: left image feature, [W,HN,C]
        :param feat_right: right image feature, [W,HN,C]
        :param pos: pos encoding, [2W-1,HN,C]
        :param pos_indexes: indexes to slicer pos encoding [W,W]
        :param last_layer: Boolean indicating if the current layer is the last layer
        :return: update image feature and attention weight
        """
        # flatten NxCxHxW to WxHNxC
        bs, c, hn, w = feat_left.shape

        feat_left = feat_left.permute(1, 3, 2, 0).flatten(2).permute(1, 2, 0)  # CxWxHxN -> CxWxHN -> WxHNxC
        feat_right = feat_right.permute(1, 3, 2, 0).flatten(2).permute(1, 2, 0)

        # concatenate left and right features
        feat_left_2 = self.norm1(feat_left)
        feat_right_2 = self.norm1(feat_right)

        # torch.save(torch.cat([feat_left_2, feat_right_2], dim=1), 'feat_cross_attn_input_' + str(layer_idx) + '.dat')

        # update right features
        if pos is not None:
            pos_flipped = torch.flip(pos, [0])
        else:
            pos_flipped = pos
        feat_right_2 = self.cross_attn(query=feat_right_2, key=feat_left_2, value=feat_left_2, pos_enc=pos_flipped,
                                       pos_indexes=pos_indexes)[0]

        feat_right = feat_right + feat_right_2

        # update left features
        # use attn mask for last layer
        if last_layer:
            w = feat_left_2.size(0)
            attn_mask = self._generate_square_subsequent_mask(w).to(feat_left.device)  # generate attn mask
        else:
            attn_mask = None

        # normalize again the updated right features
        feat_right_2 = self.norm2(feat_right)
        feat_left_2, attn_weight, raw_attn = self.cross_attn(query=feat_left_2, key=feat_right_2, value=feat_right_2,
                                                             attn_mask=attn_mask, pos_enc=pos,
                                                             pos_indexes=pos_indexes)

        # torch.save(attn_weight, 'cross_attn_' + str(layer_idx) + '.dat')

        feat_left = feat_left + feat_left_2

        # concat features
        # feat = torch.cat([feat_left, feat_right], dim=1)  # Wx2HNxC
        raw_attn = raw_attn.view(hn, bs, w, w).permute(1, 0, 2, 3)
        out_left = feat_left.view(w, hn, bs, c).permute(2, 3, 1, 0)
        out_right = feat_right.view(w, hn, bs, c).permute(2, 3, 1, 0)

        # return feat, raw_attn
        return out_left, out_right, raw_attn

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
        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        # self.global_attn = MultiheadAttn(d_model=c, nhead=4)

    def forward(self, x_l, x_r):
        b, c, h, w = x_l.shape
        # b, c, h, w = x_l.shape

        feat_l = self.l_proj1(self.norm_l(x_l))
        feat_r = self.r_proj1(self.norm_r(x_r))

        # epipolar attention
        Q_l = feat_l.permute(0, 2, 3, 1)  # B, H, Wl, c
        Q_r_T = feat_r.permute(0, 2, 1, 3) # B, H, c, Wr (transposed)

        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)  # B, H, Wl, c
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)  # B, H, Wr, c
        
        # (B, H, Wl, c) x (B, H, c, Wr) -> (B, H, Wl, Wr)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale
        
        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  #B, H, Wl, c
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l) #B, H, Wr, c

        raw_attn = [attention.contiguous(), attention.permute(0, 1, 3, 2).contiguous()]
        # scale
        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma

        return x_l + F_r2l, x_r + F_l2r, raw_attn
    

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
        # self.fusion = SCAM(c).cuda() if fusion else None
        self.fusion = CrossAttention(c, nhead = 8)
        # self.fusion = CrossFusion(d_model = c, nhead = 8).cuda() if fusion else None

    def forward(self, x_left, x_right, pos_enc, pos_indexes,last_layer=False):
        feat_left = self.blk(x_left)
        feat_right = self.blk(x_right)
        last_layer = last_layer
        if self.fusion:
            feat_left, feat_right, attn = self.fusion(feat_left, feat_right, pos_enc, pos_indexes,  last_layer)
        return feat_left, feat_right, attn

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
        self.pos_encoder = PositionEncodingSine1DRelative(args.width, normalize=False)
        self.num_blks = args.num_blks
        self.intro = nn.Conv2d(in_channels=args.img_channel, out_channels=args.width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
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
        if args.no_upsampling:
            self.out_dim = self.width
        else:
            self.out_dim = 3

    def forward(self, x):
        x_size = (x.shape[2], x.shape[3])
        feat = self.intro(x)
        feat_left, feat_right = feat.chunk(2, dim=0)
        shallow_feat_l, shallow_feat_r = feat_left, feat_right
        b, c, h, w = feat_left.shape
        # cost = torch.zeros((b,h,w,w)).to(feat_left.device)
        pos_enc = self.pos_encoder(feat_left)

        if pos_enc is not None:
            with torch.no_grad():
                # indexes to shift rel pos encoding
                indexes_r = torch.linspace(w - 1, 0, w).view(w, 1).to(feat_left.device)
                indexes_c = torch.linspace(0, w - 1, w).view(1, w).to(feat_left.device)
                pos_indexes = (indexes_r + indexes_c).view(-1).long()  # WxW' -> WW'
        else:
            pos_indexes = None
        
        for i in range(len(self.body)):
            last_layer = i == len(self.body) - 1
            feat_left, feat_right, attn = self.body[i](feat_left, feat_right, pos_enc, pos_indexes, last_layer)
            # cost = cost + attn
        
        feat_left = feat_left + shallow_feat_l
        feat_right = feat_right + shallow_feat_r
        
        return feat_left, feat_right, attn


@register('nafnet')
def make_nafnet(up_scale=4, width=48, num_blks=16, img_channel=3, drop_path_rate=0., drop_out_rate=0., fusion_from=-1, fusion_to=32, dual=True, no_upsampling =True):
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



