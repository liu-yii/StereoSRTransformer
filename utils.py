# modified from: https://github.com/yinboc/liif

import os
import time
import shutil
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import SGD, Adam
from tensorboardX import SummaryWriter
from basicsr.metrics.metric_util import reorder_image, to_y_channel
from skimage.metrics import structural_similarity
import cv2

from torchvision import transforms

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def resize_fn(img, size):
    ### normalize and resize
    transformer = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(__imagenet_stats['mean'], __imagenet_stats['std'])
    ])
    return transformer(img)

def denormalize(img):
    """
    De-normalize a tensor and return img

    :param img: normalized image, [C,H,W]
    :return: original image, [H,W,C]
    """

    if isinstance(img, torch.Tensor):
        # img = img.permute(1, 2, 0)  # H,W,C
        img *= torch.tensor(__imagenet_stats['std']).to(img.device)
        img += torch.tensor(__imagenet_stats['mean']).to(img.device)
        return img
    else:
        img = img.transpose(1, 2, 0)  # H,W,C
        img *= np.array(__imagenet_stats['std'])
        img += np.array(__imagenet_stats['mean'])
        return img

class NestedTensor(object):
    def __init__(self, left, right, disp=None, sampled_cols=None, sampled_rows=None, occ_mask=None,
                 occ_mask_right=None):
        self.left = left
        self.right = right
        self.disp = disp
        self.occ_mask = occ_mask
        self.occ_mask_right = occ_mask_right
        self.sampled_cols = sampled_cols
        self.sampled_rows = sampled_rows


def batched_index_select(source, dim, index):
    views = [source.shape[0]] + [1 if i != dim else -1 for i in range(1, len(source.shape))]
    expanse = list(source.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(source, dim, index)


def torch_1d_sample(source, sample_points, mode='linear'):
    """
    linearly sample source tensor along the last dimension
    input:
        source [N,D1,D2,D3...,Dn]
        sample_points [N,D1,D2,....,Dn-1,1]
    output:
        [N,D1,D2...,Dn-1]
    """
    idx_l = torch.floor(sample_points).long().clamp(0, source.size(-1) - 1)
    idx_r = torch.ceil(sample_points).long().clamp(0, source.size(-1) - 1)

    if mode == 'linear':
        weight_r = sample_points - idx_l
        weight_l = 1 - weight_r
    elif mode == 'sum':
        weight_r = (idx_r != idx_l).int()  # we only sum places of non-integer locations
        weight_l = 1
    else:
        raise Exception('mode not recognized')

    out = torch.gather(source, -1, idx_l) * weight_l + torch.gather(source, -1, idx_r) * weight_r
    return out.squeeze(-1)

def find_occ_mask(disp_left, disp_right):
    """
    find occlusion map
    1 indicates occlusion
    disp range [0,w]
    """
    w = disp_left.shape[-1]

    # # left occlusion
    # find corresponding pixels in target image
    coord = np.linspace(0, w - 1, w)[None,]  # 1xW
    right_shifted = coord - disp_left

    # 1. negative locations will be occlusion
    occ_mask_l = right_shifted <= 0

    # 2. wrong matches will be occlusion
    right_shifted[occ_mask_l] = 0  # set negative locations to 0
    right_shifted = right_shifted.astype(np.int)
    disp_right_selected = np.take_along_axis(disp_right, right_shifted,
                                             axis=1)  # find tgt disparity at src-shifted locations
    wrong_matches = np.abs(disp_right_selected - disp_left) > 1  # theoretically, these two should match perfectly
    wrong_matches[disp_right_selected <= 0.0] = False
    wrong_matches[disp_left <= 0.0] = False

    # produce final occ
    wrong_matches[occ_mask_l] = True  # apply case 1 occlusion to case 2
    occ_mask_l = wrong_matches

    # # right occlusion
    # find corresponding pixels in target image
    coord = np.linspace(0, w - 1, w)[None,]  # 1xW
    left_shifted = coord + disp_right

    # 1. negative locations will be occlusion
    occ_mask_r = left_shifted >= w

    # 2. wrong matches will be occlusion
    left_shifted[occ_mask_r] = 0  # set negative locations to 0
    left_shifted = left_shifted.astype(np.int)
    disp_left_selected = np.take_along_axis(disp_left, left_shifted,
                                            axis=1)  # find tgt disparity at src-shifted locations
    wrong_matches = np.abs(disp_left_selected - disp_right) > 1  # theoretically, these two should match perfectly
    wrong_matches[disp_left_selected <= 0.0] = False
    wrong_matches[disp_right <= 0.0] = False

    # produce final occ
    wrong_matches[occ_mask_r] = True  # apply case 1 occlusion to case 2
    occ_mask_r = wrong_matches

    return occ_mask_l, occ_mask_r


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:])
    rgb = img.view(3, -1).permute(1, 0)
    return coord, rgb

def loss_disp_smoothness(disp, img, img_size=None):
    if img_size is not None:
        img = img.permute(0,2,1). view(-1, 3, img_size[0], img_size[1])
        disp = disp.permute(0,2,1). view(-1, 1, img_size[0], img_size[1])
    b, _, h, w = img.size()
    img_grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
    img_grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
    weight_x = torch.exp(-torch.abs(img_grad_x).mean(1).unsqueeze(1))
    weight_y = torch.exp(-torch.abs(img_grad_y).mean(1).unsqueeze(1))

    loss = (((disp[:, :, :, :-1] - disp[:, :, :, 1:]).abs() * weight_x).sum() +
            ((disp[:, :, :-1, :] - disp[:, :, 1:, :]).abs() * weight_y).sum()) / \
           (weight_x.sum() + weight_y.sum())

    return loss


def warp(img, disp, img_size=None, mode='l2r'):
    '''
    Borrowed from: https://github.com/OniroAI/MonoDepth-PyTorch
    '''
    if img_size is not None:
        img = img.permute(0,2,1). view(-1, 3, img_size[0], img_size[1])
        disp = disp.permute(0,2,1). view(-1, 1, img_size[0], img_size[1])
    b, _, h, w = img.size()

    # Original coordinates of pixels
    x_base = torch.linspace(0, 2, w).repeat(b, h, 1).type_as(img)
    y_base = torch.linspace(0, 2, h).repeat(b, w, 1).transpose(1, 2).type_as(img)

    x_shifts = disp[:, 0, :, :]
    if mode == 'r2l':
        x_shifts = -x_shifts
    x_base = x_base + x_shifts

    grid = torch.stack((x_base, y_base), dim=1)
    shi = torch.ones_like(grid)
    grid = (grid-shi).clamp_(-1, 1)
    # grid[:,0,:,:] = 2.0*grid[:,0,:,:].clone()/max(w-1,1)-1.0 
    
    # grid[:,1,:,:] = 2.0*grid[:,1,:,:].clone()/max(h-1,1)-1.0 

    grid = grid.permute(0,2,3,1)#from B,2,H,W -> B,H,W,2
    # Apply shift in X direction
    
    mask = torch.autograd.Variable(torch.ones(img.size())).to(img.device)
    mask = F.grid_sample(mask, grid, mode='bilinear', padding_mode='border')

    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(img, grid, mode='bilinear', padding_mode='border')
    output = (output * mask).view(b, -1, h*w).permute(0, 2, 1)
    return output

def warp_coord(coord, disp, raw_hr, mask=None, mode='r2l'):
    '''
    Borrowed from:
    '''
    b, c, h, w = raw_hr.shape
    # y_disp = torch.zeros_like(disp)
    # disp = torch.cat((y_disp, disp * 2.0 / w), dim=-1)
    # if mode == 'l2r':
    #     disp = -disp
    # coord_warp = coord - disp        # b, h*w, 2
    coord_warp = torch.cat((coord[:,:,0].unsqueeze(-1), disp * 2.0 / w - 1.0), dim=-1)
    coord_warp = coord_warp.clamp_(-1, 1)
    grid = coord_warp.flip(-1).unsqueeze(1)
    # mask = torch.autograd.Variable(torch.ones(raw_hr.size())).to(raw_hr.device)
    # mask = F.grid_sample(mask, grid, mode='bilinear', padding_mode='border').permute(0, 2, 3, 1).squeeze(1)

    ret = F.grid_sample(raw_hr, grid, mode='nearest', padding_mode='border',align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
    # if mask is not None:
    #     output = ret * mask
    # else:
    #     output = ret
    output = ret

    return output


# def dispwarpfeature(feat, disp):
#     bs, channels, height, width = feat.size()
#     mh,_ = torch.meshgrid([torch.arange(0, height, dtype=feat.dtype, device=feat.device), torch.arange(0, width, dtype=feat.dtype, device=feat.device)])  # (H *W)
#     mh = mh.reshape(1, 1, height, width).repeat(bs, 1, 1, 1)

#     cur_disp_coords_y = mh
#     cur_disp_coords_x = disp

#     coords_x = cur_disp_coords_x / ((width - 1.0) / 2.0) - 1.0  # trans to -1 - 1
#     coords_y = cur_disp_coords_y / ((height - 1.0) / 2.0) - 1.0
#     grid = torch.stack([coords_x, coords_y], dim=4).view(bs, height, width, 2)   #(B, D, H, W, 2)->(B, D*H, W, 2)

#     #warped = F.grid_sample(feat, grid.view(bs, ndisp * height, width, 2), mode='bilinear', padding_mode='zeros').view(bs, channels, ndisp, height, width) 
#     warped_feat = F.grid_sample(feat, grid, mode='bilinear', padding_mode='zeros').view(bs, channels,height, width) 

#     return warped_feat


# def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
#     diff = (sr - hr) / rgb_range
#     if dataset is not None:
#         if dataset == 'benchmark':
#             shave = scale
#             if diff.size(1) > 1:
#                 gray_coeffs = [65.738, 129.057, 25.064]
#                 convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
#                 diff = diff.mul(convert).sum(dim=1)
#         elif dataset == 'div2k':
#             shave = scale + 6
#         elif dataset == 'AID':
#             shave = scale + 6
#         else:
#             raise NotImplementedError
#         valid = diff[..., shave:-shave, shave:-shave]
#     else:
#         valid = diff
#     mse = valid.pow(2).mean()
#     return -10 * torch.log10(mse)

# def calc_ssim(sr, hr, dataset=None, scale=1, rgb_range=1):
#     diff = (sr - hr) / rgb_range
#     if dataset is not None:
#         if dataset == 'benchmark':
#             shave = scale
#             if diff.size(1) > 1:
#                 gray_coeffs = [65.738, 129.057, 25.064]
#                 convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
#                 diff = diff.mul(convert).sum(dim=1)
#         elif dataset == 'div2k':
#             shave = scale + 6
#         elif dataset == 'AID':
#             shave = scale + 6
#         else:
#             raise NotImplementedError
#         valid = diff[..., shave:-shave, shave:-shave]
#     else:
#         valid = diff
#     mse = valid.pow(2).mean()
#     return -10 * torch.log10(mse)

def calculate_psnr(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray/tensor): Images with range [0, 255]/[0, 1].
        img2 (ndarray/tensor): Images with range [0, 255]/[0, 1].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    if type(img1) == torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1,2,0)
    if type(img2) == torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1,2,0)
        
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
    
    def _psnr(img1, img2):
        if test_y_channel:
            img1 = to_y_channel(img1)
            img2 = to_y_channel(img2)

        mse = np.mean((img1 - img2)**2)
        if mse == 0:
            return float('inf')
        max_value = 1. if img1.max() <= 1 else 255.
        return 20. * np.log10(max_value / np.sqrt(mse))
    
    if img1.ndim == 3 and img1.shape[2] == 6:
        l1, r1 = img1[:,:,:3], img1[:,:,3:]
        l2, r2 = img2[:,:,:3], img2[:,:,3:]
        return (_psnr(l1, l2) + _psnr(r1, r2))/2
    else:
        return _psnr(img1, img2)

def calculate_psnr_left(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):
    assert input_order == 'HWC'
    assert crop_border == 0

    img1 = img1[:,64:,:3]
    img2 = img2[:,64:,:3]
    return calculate_psnr(img1=img1, img2=img2, crop_border=0, input_order=input_order, test_y_channel=test_y_channel)

def _ssim(img1, img2, max_value):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * max_value)**2
    C2 = (0.03 * max_value)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def prepare_for_ssim(img, k):
    import torch
    with torch.no_grad():
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
        conv = torch.nn.Conv2d(1, 1, k, stride=1, padding=k//2, padding_mode='reflect')
        conv.weight.requires_grad = False
        conv.weight[:, :, :, :] = 1. / (k * k)

        img = conv(img)

        img = img.squeeze(0).squeeze(0)
        img = img[0::k, 0::k]
    return img.detach().cpu().numpy()

def prepare_for_ssim_rgb(img, k):
    import torch
    with torch.no_grad():
        img = torch.from_numpy(img).float() #HxWx3

        conv = torch.nn.Conv2d(1, 1, k, stride=1, padding=k // 2, padding_mode='reflect')
        conv.weight.requires_grad = False
        conv.weight[:, :, :, :] = 1. / (k * k)

        new_img = []

        for i in range(3):
            new_img.append(conv(img[:, :, i].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)[0::k, 0::k])

    return torch.stack(new_img, dim=2).detach().cpu().numpy()

def _3d_gaussian_calculator(img, conv3d):
    out = conv3d(img.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    return out

def _generate_3d_gaussian_kernel():
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    kernel_3 = cv2.getGaussianKernel(11, 1.5)
    kernel = torch.tensor(np.stack([window * k for k in kernel_3], axis=0))
    conv3d = torch.nn.Conv3d(1, 1, (11, 11, 11), stride=1, padding=(5, 5, 5), bias=False, padding_mode='replicate')
    conv3d.weight.requires_grad = False
    conv3d.weight[0, 0, :, :, :] = kernel
    return conv3d

def _ssim_3d(img1, img2, max_value):
    assert len(img1.shape) == 3 and len(img2.shape) == 3
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255]/[0, 1] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255]/[0, 1] with order 'HWC'.

    Returns:
        float: ssim result.
    """
    C1 = (0.01 * max_value) ** 2
    C2 = (0.03 * max_value) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = _generate_3d_gaussian_kernel().cuda()

    img1 = torch.tensor(img1).float().cuda()
    img2 = torch.tensor(img2).float().cuda()


    mu1 = _3d_gaussian_calculator(img1, kernel)
    mu2 = _3d_gaussian_calculator(img2, kernel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = _3d_gaussian_calculator(img1 ** 2, kernel) - mu1_sq
    sigma2_sq = _3d_gaussian_calculator(img2 ** 2, kernel) - mu2_sq
    sigma12 = _3d_gaussian_calculator(img1*img2, kernel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())

def _ssim_cly(img1, img2):
    assert len(img1.shape) == 2 and len(img2.shape) == 2
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    # print(kernel)
    window = np.outer(kernel, kernel.transpose())

    bt = cv2.BORDER_REPLICATE

    mu1 = cv2.filter2D(img1, -1, window, borderType=bt)
    mu2 = cv2.filter2D(img2, -1, window,borderType=bt)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window, borderType=bt) - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window, borderType=bt) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window, borderType=bt) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False,
                   ssim3d=True):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')

    if type(img1) == torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1,2,0)
    if type(img2) == torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1,2,0)

    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    def _cal_ssim(img1, img2):
        if test_y_channel:
            img1 = to_y_channel(img1)
            img2 = to_y_channel(img2)
            return _ssim_cly(img1[..., 0], img2[..., 0])

        ssims = []
        # ssims_before = []

        # skimage_before = skimage.metrics.structural_similarity(img1, img2, data_range=255., multichannel=True)
        # print('.._skimage',
        #       skimage.metrics.structural_similarity(img1, img2, data_range=255., multichannel=True))
        max_value = 1 if img1.max() <= 1 else 255
        with torch.no_grad():
            final_ssim = _ssim_3d(img1, img2, max_value) if ssim3d else _ssim(img1, img2, max_value)
            ssims.append(final_ssim)

        # for i in range(img1.shape[2]):
        #     ssims_before.append(_ssim(img1, img2))

        # print('..ssim mean , new {:.4f}  and before {:.4f} .... skimage before {:.4f}'.format(np.array(ssims).mean(), np.array(ssims_before).mean(), skimage_before))
            # ssims.append(skimage.metrics.structural_similarity(img1[..., i], img2[..., i], multichannel=False))

        return np.array(ssims).mean()

    if img1.ndim == 3 and img1.shape[2] == 6:
        l1, r1 = img1[:,:,:3], img1[:,:,3:]
        l2, r2 = img2[:,:,:3], img2[:,:,3:]
        return (_cal_ssim(l1, l2) + _cal_ssim(r1, r2))/2
    else:
        return _cal_ssim(img1, img2)

def calculate_ssim_left(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False,
                   ssim3d=True):
    assert input_order == 'HWC'
    assert crop_border == 0

    img1 = img1[:,64:,:3]
    img2 = img2[:,64:,:3]
    return calculate_ssim(img1=img1, img2=img2, crop_border=0, input_order=input_order, test_y_channel=test_y_channel, ssim3d=ssim3d)

def calculate_skimage_ssim(img1, img2):
    return structural_similarity(img1, img2, channel_axis=2)

def calculate_skimage_ssim_left(img1, img2):
    img1 = img1[:,64:,:3]
    img2 = img2[:,64:,:3]
    return calculate_skimage_ssim(img1=img1, img2=img2)
