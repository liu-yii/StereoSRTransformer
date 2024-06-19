import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from utils import to_pixel_samples
from utils import make_coord, normalize, resize_fn
   

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
    
    def __getitem__(self, idx):
        imgl = self.dataset[idx]['img_l']
        imgr = self.dataset[idx]['img_r']
        # img = torch.concatenate([imgl,imgr], dim=-1)
        filename = self.dataset[idx]['filename']
        disp = self.dataset[idx]['disp']

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
           
            imgl = imgl[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
            imgl_down = resize_fn(imgl, (h_lr, w_lr))
            cropl_lr, cropl_hr = imgl_down, imgl
            imgr = imgr[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
            imgr_down = resize_fn(imgr, (h_lr, w_lr))
            cropr_lr, cropr_hr = imgr_down, imgr
            disp = disp[:, :round(h_lr * s), :round(w_lr * s)]
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
            disp = disp[:, x0: x0 + h_hr, y0: y0 + w_hr]
        
        cropl_hr = normalize(cropl_hr)
        cropr_hr = normalize(cropr_hr)
        cropl_lr = normalize(cropl_lr)
        cropr_lr = normalize(cropr_lr)

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
            disp = augment(disp)
        
        hr_coord, hrl_rgb = to_pixel_samples(cropl_hr.contiguous())
        _, hrr_rgb = to_pixel_samples(cropr_hr.contiguous())
        
        if self.sample_q is not None:
            sample_lst = np.random.choice(
                min(len(hrl_rgb),len(hrr_rgb)), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / cropl_hr.shape[-2]
        cell[:, 1] *= 2 / cropl_hr.shape[-1]
        crop_lr = torch.concatenate([cropl_lr,cropr_lr],dim=0)  #[6, h, w]
        hr_rgb = torch.concatenate([cropl_hr,cropr_hr],dim=0)    #[6, H, W]
        return {
            'inp': crop_lr,   #[6, h, w]
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,  #[6, H, W]
            'disp': disp,    #[1, H, W]
        }, filename


@register('sr-implicit-stereo-without-disp')
class SRImplicitStereoTest(Dataset):

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

    def __getitem__(self, idx):
        imgl = self.dataset[idx]['img_l']
        imgr = self.dataset[idx]['img_r']
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
        
        cropl_hr = normalize(cropl_hr)
        cropr_hr = normalize(cropr_hr)
        cropl_lr = normalize(cropl_lr)
        cropr_lr = normalize(cropr_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            # dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                return x

            cropl_lr = augment(cropl_lr)
            cropl_hr = augment(cropl_hr)
            cropr_lr = augment(cropr_lr)
            cropr_hr = augment(cropr_hr)
        
        hr_coord, hrl_rgb = to_pixel_samples(cropl_hr.contiguous())
        _, hrr_rgb = to_pixel_samples(cropr_hr.contiguous())
        

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                min(len(hrl_rgb),len(hrr_rgb)), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / cropl_hr.shape[-2]
        cell[:, 1] *= 2 / cropl_hr.shape[-1]
        crop_lr = torch.concatenate([cropl_lr,cropr_lr],dim=0)  #[3,H,2W]
        hr_rgb = torch.concatenate([cropl_hr, cropr_hr],dim=0)    #[2304,6]
        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }, filename