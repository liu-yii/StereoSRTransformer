import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from .ssrtr import LayerNorm2d
from functools import partial
import timm
from timm.models.layers import DropPath, trunc_normal_

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
        
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MultiscaleBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn_multiscale = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        '''
        Multi-level aggregation
        '''
        # x: [B, H, U, V]
        B, H, U, V = x.shape
        if H == 1:
            x = x.flatten(0, 1)
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x.view(B, H, U, V)
        x = x.flatten(0, 1)    # [B*H, U, V]
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp2(self.norm4(x)))
        x = x.view(B, H, U, V).transpose(1, 2).flatten(0, 1)  # [B*U, H, V]
        x = x + self.drop_path(self.attn_multiscale(self.norm3(x)))
        x = x.view(B, U, H, V).transpose(1, 2).flatten(0, 1)  # [B*H, U, V]
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.view(B, H, U, V)
        return x
    
class AggregationTransformer(nn.Module):
    """
    Aggregation transformer
    """

    def __init__(self, img_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None):
        super().__init__()
        self.img_h, self.img_w = img_size
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.img_h, self.img_w, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            MultiscaleBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.proj = nn.Linear(embed_dim, self.img_w)
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        # trunc_normal_(self.pos_embed_y, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, corr, feat_l, feat_r):
        '''
        :param corr: correlation matrix [B, H, U, V]
        :param feat_l: feature map of left image [B, H, U, proj_dim]
        :param feat_r: feature map of right image [B, H, V, proj_dim]
        '''
        
        B = corr.shape[0] 
        x = corr.clone() 
        
        pos_embed = self.pos_embed

        x = torch.cat((x.transpose(-1, -2), feat_r), dim=3) + pos_embed
        x = self.proj(self.blocks(x)).transpose(-1, -2) + corr  # swapping the axis for swapping self-attention.
        x = torch.cat((x, feat_l), dim=3) + pos_embed
        x = self.proj(self.blocks(x)) + corr 

        return x
    

class CAT(nn.Module):
    def __init__(self, feature_size, proj_dim, depth = 2, num_heads = 4, mlp_ratio = 4):
        super(CAT, self).__init__()
        self.feature_h, self.feature_w = feature_size
        self.proj_dim = proj_dim
        self.decoder_embed_dim = self.feature_w + self.proj_dim
        self.proj_l = nn.Linear(48, proj_dim)
        self.proj_r = nn.Linear(48, proj_dim)
        self.decoder = AggregationTransformer(img_size=feature_size, 
                                              embed_dim=self.decoder_embed_dim, 
                                              depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                                              qkv_bias=True, qk_scale=None,)
        self.x_normal = np.linspace(-1,1,self.feature_w)
        self.x_normal = nn.Parameter(torch.tensor(self.x_normal, dtype=torch.float, requires_grad=False))

    def softmax_with_temperature(self, x, beta, d = 1):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        M, _ = x.max(dim=d, keepdim=True)
        x = x - M # subtract maximum value for stability
        exp_x = torch.exp(x/beta)
        exp_x_sum = exp_x.sum(dim=d, keepdim=True)
        return exp_x / exp_x_sum

    def soft_argmax(self, corr, beta=0.02):
        r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
        b,h,w,w = corr.shape
        corr = self.softmax_with_temperature(corr, beta, d = -1)
        index = torch.arange(w).view(1, 1, 1, w).to(corr.device).float()    # index: 1*1*1*w
        coordx = torch.arange(w).view(1, 1, 1, w).repeat(1,1,h,1).to(corr.device).float() 
        disp = coordx.squeeze(0) - torch.sum(corr*index, dim=-1).view(b, 1, h, w) # x axis of the corresponding point

        return disp
        
    
    def mutual_nn_filter(self, correlation_matrix):
        r"""Mutual nearest neighbor filtering (Rocco et al. NeurIPS'18)"""
        corr_src_max = torch.max(correlation_matrix, dim=3, keepdim=True)[0]
        corr_trg_max = torch.max(correlation_matrix, dim=2, keepdim=True)[0]
        corr_src_max[corr_src_max == 0] += 1e-30
        corr_trg_max[corr_trg_max == 0] += 1e-30

        corr_src = correlation_matrix / corr_src_max
        corr_trg = correlation_matrix / corr_trg_max

        return correlation_matrix * (corr_src * corr_trg)
    
    def forward(self, attn_weight, feat_l, feat_r):
        corr, _ = attn_weight
        corr = self.mutual_nn_filter(corr)
        feat_l_proj = self.proj_l(feat_l.permute(0, 2, 3, 1).contiguous())
        feat_r_proj = self.proj_r(feat_r.permute(0, 2, 3, 1).contiguous())
        refined_corr = self.decoder(corr, feat_l_proj, feat_r_proj)
        refined_corr = corr
        disp = self.soft_argmax(refined_corr)
        # disp = unnormalise_and_convert_mapping_to_flow(grid)
        return disp
    


