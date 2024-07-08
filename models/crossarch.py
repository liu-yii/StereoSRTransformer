import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from utils import get_clones

class MultiheadAttentionRelative(nn.MultiheadAttention):
    """
    Multihead attention with relative positional encoding
    """

    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttentionRelative, self).__init__(embed_dim, num_heads, dropout=0.0, bias=True,
                                                         add_bias_kv=False, add_zero_attn=False,
                                                         kdim=None, vdim=None)

    def forward(self, query, key, value, attn_mask=None, pos_enc=None, pos_indexes=None):
        """
        Multihead attention

        :param query: [W,HN,C]
        :param key: [W,HN,C]
        :param value: [W,HN,C]
        :param attn_mask: mask to invalidate attention, -inf is used for invalid attention, [W,W]
        :param pos_enc: [2W-1,C]
        :param pos_indexes: index to select relative encodings, flattened in transformer WW
        :return: output value vector, attention with softmax (for debugging) and raw attention (used for last layer)
        """

        w, bsz, embed_dim = query.size()
        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # project to get qkv
        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)

        elif torch.equal(key, value):
            # cross-attention
            _b = self.in_proj_bias
            _start = 0
            _end = embed_dim
            _w = self.in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:
                _b = self.in_proj_bias
                _start = embed_dim
                _end = None
                _w = self.in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

        # project to find q_r, k_r
        if pos_enc is not None:
            # reshape pos_enc
            pos_enc = torch.index_select(pos_enc, 0, pos_indexes).view(w, w,
                                                                       -1)  # 2W-1xC -> WW'xC -> WxW'xC
            # compute k_r, q_r
            _start = 0
            _end = 2 * embed_dim
            _w = self.in_proj_weight[_start:_end, :]
            _b = self.in_proj_bias[_start:_end]
            q_r, k_r = F.linear(pos_enc, _w, _b).chunk(2, dim=-1)  # WxW'xC
        else:
            q_r = None
            k_r = None

        # scale query
        scaling = float(head_dim) ** -0.5
        q = q * scaling
        if q_r is not None:
            q_r = q_r * scaling

        # reshape
        q = q.contiguous().view(w, bsz, self.num_heads, head_dim)  # WxNxExC
        if k is not None:
            k = k.contiguous().view(-1, bsz, self.num_heads, head_dim)
        if v is not None:
            v = v.contiguous().view(-1, bsz, self.num_heads, head_dim)

        if q_r is not None:
            q_r = q_r.contiguous().view(w, w, self.num_heads, head_dim)  # WxW'xExC
        if k_r is not None:
            k_r = k_r.contiguous().view(w, w, self.num_heads, head_dim)

        # compute attn weight
        attn_feat = torch.einsum('wnec,vnec->newv', q, k)  # NxExWxW'

        # add positional terms
        if pos_enc is not None:
            # 0.3 s
            attn_feat_pos = torch.einsum('wnec,wvec->newv', q, k_r)  # NxExWxW'
            attn_pos_feat = torch.einsum('vnec,wvec->newv', k, q_r)  # NxExWxW'

            # 0.1 s
            attn = attn_feat + attn_feat_pos + attn_pos_feat
        else:
            attn = attn_feat

        assert list(attn.size()) == [bsz, self.num_heads, w, w]

        # apply attn mask
        if attn_mask is not None:
            attn_mask = attn_mask[None, None, ...]
            attn += attn_mask

        # raw attn
        raw_attn = attn

        # softmax
        attn = F.softmax(attn, dim=-1)

        # compute v, equivalent to einsum('',attn,v),
        # need to do this because apex does not support einsum when precision is mixed
        v_o = torch.bmm(attn.view(bsz * self.num_heads, w, w),
                        v.permute(1, 2, 0, 3).view(bsz * self.num_heads, w, head_dim))  # NxExWxW', W'xNxExC -> NExWxC
        assert list(v_o.size()) == [bsz * self.num_heads, w, head_dim]
        v_o = v_o.reshape(bsz, self.num_heads, w, head_dim).permute(2, 0, 1, 3).reshape(w, bsz, embed_dim)
        v_o = F.linear(v_o, self.out_proj.weight, self.out_proj.bias)

        # average attention weights over heads
        attn = attn.sum(dim=1) / self.num_heads

        # raw attn
        raw_attn = raw_attn.sum(dim=1)

        return v_o, attn, raw_attn

class Transformer(nn.Module):
    """
    Transformer computes self (intra image) and cross (inter image) attention
    """

    def __init__(self, hidden_dim: int = 128, nhead: int = 8, num_attn_layers: int = 4):
        super().__init__()

        self_attn_layer = TransformerSelfAttnLayer(hidden_dim, nhead)
        self.self_attn_layers = get_clones(self_attn_layer, num_attn_layers)

        cross_attn_layer = TransformerCrossAttnLayer(hidden_dim, nhead)
        self.cross_attn_layers = get_clones(cross_attn_layer, num_attn_layers)

        self.norm = nn.LayerNorm(hidden_dim)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_attn_layers = num_attn_layers

    def _alternating_attn(self, feat, pos_enc, pos_indexes, hn):
        """
        Alternate self and cross attention with gradient checkpointing to save memory

        :param feat: image feature concatenated from left and right, [W,2HN,C]
        :param pos_enc: positional encoding, [W,HN,C]
        :param pos_indexes: indexes to slice positional encoding, [W,HN,C]
        :param hn: size of HN
        :return: attention weight [N,H,W,W]
        """

        global layer_idx
        # alternating
        for idx, (self_attn, cross_attn) in enumerate(zip(self.self_attn_layers, self.cross_attn_layers)):
            layer_idx = idx

            # checkpoint self attn
            def create_custom_self_attn(module):
                def custom_self_attn(*inputs):
                    return module(*inputs)

                return custom_self_attn

            feat = checkpoint(create_custom_self_attn(self_attn), feat, pos_enc, pos_indexes)

            # add a flag for last layer of cross attention
            if idx == self.num_attn_layers - 1:
                # checkpoint cross attn
                def create_custom_cross_attn(module):
                    def custom_cross_attn(*inputs):
                        return module(*inputs, True)

                    return custom_cross_attn
            else:
                # checkpoint cross attn
                def create_custom_cross_attn(module):
                    def custom_cross_attn(*inputs):
                        return module(*inputs, False)

                    return custom_cross_attn

            feat, attn_weight = checkpoint(create_custom_cross_attn(cross_attn), feat[:, :hn], feat[:, hn:], pos_enc,
                                           pos_indexes)

        layer_idx = 0
        return attn_weight, feat

    def forward(self, feat_left, feat_right, pos_enc = None):
        """
        :param feat_left: feature descriptor of left image, [N,C,H,W]
        :param feat_right: feature descriptor of right image, [N,C,H,W]
        :param pos_enc: relative positional encoding, [N,C,H,2W-1]
        :return: cross attention values [N,H,W,W], dim=2 is left image, dim=3 is right image
        """

        # flatten NxCxHxW to WxHNxC
        bs, c, hn, w = feat_left.shape

        feat_left = feat_left.permute(1, 3, 2, 0).flatten(2).permute(1, 2, 0)  # CxWxHxN -> CxWxHN -> WxHNxC
        feat_right = feat_right.permute(1, 3, 2, 0).flatten(2).permute(1, 2, 0)
        if pos_enc is not None:
            with torch.no_grad():
                # indexes to shift rel pos encoding
                indexes_r = torch.linspace(w - 1, 0, w).view(w, 1).to(feat_left.device)
                indexes_c = torch.linspace(0, w - 1, w).view(1, w).to(feat_left.device)
                pos_indexes = (indexes_r + indexes_c).view(-1).long()  # WxW' -> WW'
        else:
            pos_indexes = None

        # concatenate left and right features
        feat = torch.cat([feat_left, feat_right], dim=1)  # Wx2HNxC

        # compute attention
        attn_weight, feat = self._alternating_attn(feat, pos_enc, pos_indexes, hn*bs)
        attn_weight = attn_weight.view(hn, bs, w, w).permute(1, 0, 2, 3)  # NxHxWxW, dim=2 left image, dim=3 right image
        # feat = feat.view(w, hn, bs, c).permute(2, 3, 1, 0)  # NxCxHxW
        out_left, out_right = feat.chunk(2, dim=1)
        out_left = out_left.view(w, hn, bs, c).permute(2, 3, 1, 0)  # NxCxHxW
        out_right = out_right.view(w, hn, bs, c).permute(2, 3, 1, 0)  # NxCxHxW

        return attn_weight, out_left, out_right

class TransformerSelfAttnLayer(nn.Module):
    """
    Self attention layer
    """

    def __init__(self, hidden_dim: int, nhead: int):
        super().__init__()
        self.self_attn = MultiheadAttentionRelative(hidden_dim, nhead)

        self.norm1 = nn.LayerNorm(hidden_dim)

    def forward(self, feat,
                pos = None,
                pos_indexes = None):
        """
        :param feat: image feature [W,2HN,C]
        :param pos: pos encoding [2W-1,HN,C]
        :param pos_indexes: indexes to slice pos encoding [W,W]
        :return: updated image feature
        """
        feat2 = self.norm1(feat)

        # torch.save(feat2, 'feat_self_attn_input_' + str(layer_idx) + '.dat')

        feat2, attn_weight, _ = self.self_attn(query=feat2, key=feat2, value=feat2, pos_enc=pos,
                                               pos_indexes=pos_indexes)

        # torch.save(attn_weight, 'self_attn_' + str(layer_idx) + '.dat')

        feat = feat + feat2

        return feat

class TransformerCrossAttnLayer(nn.Module):

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
        feat = torch.cat([feat_left, feat_right], dim=1)  # Wx2HNxC

        return feat, raw_attn

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
    


class dispnet(nn.Module):
    def __init__(self, hidden_dim=128, nhead=8, num_attn_layers=4, ot=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_attn_layers = num_attn_layers
        self.ot = ot
        self.transformer = Transformer(hidden_dim, nhead, num_attn_layers)
        self.phi = nn.Parameter(torch.tensor(0.0, requires_grad=True))  # dustbin cost
    
    def _compute_unscaled_pos_shift(self, w, device):
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

    def _compute_low_res_disp(self, pos_shift, attn_weight, occ_mask = None):
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

    def _sinkhorn(self, attn, log_mu, log_nu, iters):
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

    def _optimal_transport(self, attn, iters):
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

    def _compute_low_res_occ(self, matched_attn):
        """
        Compute low res occlusion by using inverse of the matched values

        :param matched_attn: updated attention weight without dustbins, [N,H,W,W]
        :return: low res occlusion map, [N,H,W]
        """
        occ_pred = 1.0 - matched_attn
        return occ_pred.squeeze(-1)
    
    def forward(self, init_corr, feat_left, feat_right, pos_enc = None):
        """
        :param feat_left: feature descriptor of left image, [N,C,H,W]
        :param feat_right: feature descriptor of right image, [N,C,H,W]
        :return: low res disparity, [N,1,H,W], left and right feature maps [N,C,H,W]
        """
        attn_weight, out_left, out_right = self.transformer(feat_left, feat_right, pos_enc)
        attn_weight = attn_weight + init_corr
         # normalize attention to 0-1
        if self.ot:
            # optimal transport
            attn_ot = self._optimal_transport(attn_weight, 10)
        else:
            # softmax
            attn_ot = self._softmax(attn_weight)

        # regress low res disparity
        pos_shift = self._compute_unscaled_pos_shift(attn_weight.shape[2], attn_weight.device)  # NxHxW_leftxW_right
        raw_disp, matched_attn = self._compute_low_res_disp(pos_shift, attn_ot[..., :-1, :-1])
        
        # regress low res occlusion
        raw_occ = self._compute_low_res_occ(matched_attn)
        raw_disp, raw_occ = raw_disp.unsqueeze(1), raw_occ.unsqueeze(1)

        return raw_disp, out_left, out_right