import functools
import random
import math
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from utils import to_pixel_samples
from utils import make_coord

@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr, filename = self.dataset[idx]
        

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
            'filename': filename
        }

@register('sr-implicit-paired-fast')
class SRImplicitPairedFast(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr, filename = self.dataset[idx]
        

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            h_hr = s * h_lr
            w_hr = s * w_lr
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord = make_coord([h_hr, w_hr], flatten=False)
        hr_rgb = crop_hr
        
        if self.inp_size is not None:
            x0 = random.randint(0, h_hr - h_lr)
            y0 = random.randint(0, w_hr - w_lr)
            
            hr_coord = hr_coord[x0:x0+self.inp_size, y0:y0+self.inp_size, :]
            hr_rgb = crop_hr[:, x0:x0+self.inp_size, y0:y0+self.inp_size]
        
        cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
            'filename': filename
        }
    
    
def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))


@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]['img']
        filename = self.dataset[idx]['filename']
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            img = img[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            w_lr = self.inp_size
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
            'filename': filename
        }


@register('sr-implicit-downsampled-fast')
class SRImplicitDownsampledFast(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            img = img[:, :h_hr, :w_hr] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            h_lr = self.inp_size
            w_lr = self.inp_size
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord = make_coord([h_hr, w_hr], flatten=False)
        hr_rgb = crop_hr
        
        if self.inp_size is not None:
            idx = torch.tensor(np.random.choice(h_hr*w_hr, h_lr*w_lr, replace=False))
            hr_coord = hr_coord.view(-1, hr_coord.shape[-1])
            hr_coord = hr_coord[idx, :]
            hr_coord = hr_coord.view(h_lr, w_lr, hr_coord.shape[-1])

            hr_rgb = crop_hr.contiguous().view(crop_hr.shape[0], -1)
            hr_rgb = hr_rgb[:, idx]
            hr_rgb = hr_rgb.view(crop_hr.shape[0], h_lr, w_lr)
        
        cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }    
    
    
@register('sr-implicit-uniform-varied')
class SRImplicitUniformVaried(Dataset):

    def __init__(self, dataset, size_min, size_max=None,
                 augment=False, gt_resize=None, sample_q=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        p = idx / (len(self.dataset) - 1)
        w_hr = round(self.size_min + (self.size_max - self.size_min) * p)
        img_hr = resize_fn(img_hr, w_hr)

        if self.augment:
            if random.random() < 0.5:
                img_lr = img_lr.flip(-1)
                img_hr = img_hr.flip(-1)

        if self.gt_resize is not None:
            img_hr = resize_fn(img_hr, self.gt_resize)

        hr_coord, hr_rgb = to_pixel_samples(img_hr)

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / img_hr.shape[-2]
        cell[:, 1] *= 2 / img_hr.shape[-1]

        return {
            'inp': img_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }
    

@register('sr-implicit-stereo')
class SRImplicitStereo(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None, window_size=8):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q
        self.window_size = window_size
        self.s = random.uniform(self.scale_min, self.scale_max)

    def __len__(self):
        return len(self.dataset)
    
    # def check_size(self, h, w):
    #     if h % self.window_size != 0:
    #         h = math.floor(h // self.window_size) * self.window_size
    #     if w % self.window_size != 0:
    #         w = math.floor(w // self.window_size) * self.window_size
    #     return h, w

    def __getitem__(self, idx):
        imgl = self.dataset[idx]['img_l']
        imgr = self.dataset[idx]['img_r']
        # img = torch.concatenate([imgl,imgr], dim=-1)
        filename = self.dataset[idx]['filename']

        s = self.s
        
        if self.inp_size is None:
            h_lr = math.floor(imgl.shape[-2] / s + 1e-9)
            w_lr = math.floor(imgl.shape[-1] / s + 1e-9)
            if self.window_size != 0:
                # SwinIR Evaluation - reflection padding
                # batch size : 1 for testing
                # h_old, w_old = imgl.shape[-2:]
                h_pad = (h_lr // self.window_size + 1) * self.window_size - h_lr
                w_pad = (w_lr // self.window_size + 1) * self.window_size - w_lr
                h_lr += h_pad
                w_lr += w_pad
                imgl = torch.cat([imgl, torch.flip(imgl, [1])], 1)[..., :round(h_lr * s), :]
                imgl = torch.cat([imgl, torch.flip(imgl, [2])], 2)[..., :round(w_lr * s)]
                imgr = torch.cat([imgr, torch.flip(imgr, [1])], 1)[..., :round(h_lr * s), :]
                imgr = torch.cat([imgr, torch.flip(imgr, [2])], 2)[..., :round(w_lr * s)]
            w_hr = round(w_lr * s)
            h_hr = round(h_lr * s)
            imgl = imgl[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
            imgl_down = resize_fn(imgl, (h_lr, w_lr))
            cropl_lr, cropl_hr = imgl_down, imgl
            imgr = imgr[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
            imgr_down = resize_fn(imgr, (h_lr, w_lr))
            cropr_lr, cropr_hr = imgr_down, imgr
        else:
            h_lr, w_lr = self.inp_size
            w_hr = round(w_lr * s)
            h_hr = round(h_lr * s)
            x0 = random.randint(0, imgl.shape[-2] - h_hr)
            y0 = random.randint(0, imgl.shape[-1] - w_hr)
            cropl_hr = imgl[:, x0: x0 + h_hr, y0: y0 + w_hr]
            cropl_lr = resize_fn(cropl_hr, (h_lr, w_lr))
            cropr_hr = imgr[:, x0: x0 + h_hr, y0: y0 + w_hr]
            cropr_lr = resize_fn(cropr_hr, (h_lr, w_lr))
        
        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            # dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                # if dflip:
                #     x = x.transpose(-2, -1)
                return x

            cropl_lr = augment(cropl_lr)
            cropl_hr = augment(cropl_hr)
            cropr_lr = augment(cropr_lr)
            cropr_hr = augment(cropr_hr)
        ## Raw HR
        raw_hrl, raw_hrr = cropl_hr, cropr_hr
        ## Coord and RGB
        hr_coord, hrl_rgb = to_pixel_samples(cropl_hr.contiguous())
        _, hrr_rgb = to_pixel_samples(cropr_hr.contiguous())
        
        
        # Sample
        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hrl_rgb = hrl_rgb[sample_lst]
            hrr_rgb = hrr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / cropl_hr.shape[-2]
        cell[:, 1] *= 2 / cropl_hr.shape[-1]
        crop_lr = torch.cat([cropl_lr,cropr_lr],dim=-1)  #[3,H,2W]
        hr_rgb = torch.cat([hrl_rgb, hrr_rgb],dim=-1)    #[2304,6]
        raw_hr = torch.cat([raw_hrl, raw_hrr],dim=-1)    #[h_max*w_max, 10]
        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
            'raw_hr': raw_hr,
        }, filename, [h_hr, w_hr]