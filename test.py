import argparse
import os
import math
from functools import partial
import matplotlib.pyplot as plt

import yaml
import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils
from utils import make_coord, denormalize
import cv2
import numpy as np
from PIL import Image
import lpips
import time



def batched_predict(model, inp_left, inp_right, coord, cell, bsize, scale):
    with torch.no_grad():
        model.gen_feat(inp_left, inp_right, scale)
        n = coord.shape[1]
        ql = 0
        preds_left = []
        preds_right = []
        preds_disp = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred_left = model.query_rgb_left(coord[:, ql: qr, :], cell[:, ql: qr, :])
            pred_right = model.query_rgb_right(coord[:, ql: qr, :], cell[:, ql: qr, :])
            pred_disp = model.query_disp(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds_left.append(pred_left)
            preds_right.append(pred_right)
            preds_disp.append(pred_disp)
            ql = qr
        pred_left = torch.cat(preds_left, dim=1)
        pred_right = torch.cat(preds_right, dim=1)
        pred_disp = torch.cat(preds_disp, dim=1)
    return pred_left, pred_right, pred_disp


def eval_psnr(loader, model, save_dir, data_norm=None, eval_type=None, eval_bsize=None, scale_max=4, fast=False,
              verbose=False):
    if save_dir != None:
        os.makedirs(save_dir, exist_ok=True)
    model.eval()

    if eval_type is None:
        metric_fn = partial(utils.calculate_psnr, crop_border=0)
        scale = 2
    elif eval_type.startswith('test'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calculate_psnr, crop_border=0, test_y_channel=False)
    else:
        raise NotImplementedError

    val_res_left = utils.Averager()
    val_res_right = utils.Averager()
    val_res_avg = utils.Averager()
    time_avg = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        data, filename = batch
        for k, v in data.items():
            if isinstance(data[k], list):
                for idx in range(len(data[k])):
                    data[k][idx] = v[idx].cuda()
            else:
                if v is not None:
                    data[k] = v.cuda()
        filename = filename[0].split('.')[0]
        inp = torch.stack(data['inp'], dim=0)
        gt = torch.stack(data['gt'], dim=0)
        coord = torch.stack(data['coord'], dim=0)
        cell = torch.stack(data['cell'], dim=0)
        inp_left, inp_right = torch.chunk(inp, 2, dim=1)
        gt_left, gt_right = torch.chunk(gt, 2, dim=1)
        # # GT color
        # gt_l_rgb = F.grid_sample(gt_left, data['coord'].flip(-1).unsqueeze(1), mode='nearest', padding_mode='border')[:, :, 0, :] \
        #             .permute(0, 2, 1)
        # gt_r_rgb = F.grid_sample(gt_right, data['coord'].flip(-1).unsqueeze(1), mode='nearest', padding_mode='border')[:, :, 0, :] \
        #             .permute(0, 2, 1)
        scale = torch.Tensor([gt_left.shape[-1] / inp_left.shape[-1]]).to(inp_left.device)

        _, _, h, w = inp_left.size()
        
        start_time = time.time()    
        if eval_bsize is None:
            with torch.no_grad():
                pred_left, pred_right, pred_disp = model(inp_left, inp_right, coord, cell, scale)
        else:
            if fast:
                pred_left, pred_right, pred_disp = model(inp_left, inp_right, coord, cell*max(scale/scale_max, 1), scale)
            else:
                pred_left, pred_right, pred_disp = batched_predict(model, inp_left, inp_right, coord, cell*max(scale/scale_max, 1), eval_bsize, scale) # cell clip for extrapolation
        infer_time = time.time()-start_time

           
        # gt reshape
        h_hr, w_hr = gt_left.shape[-2:]
       
        # # prediction reshape
        shape = [inp_left.shape[0], h_hr, w_hr, 3]
        pred_left = pred_left.view(*shape) \
            .permute(0, 3, 1, 2).contiguous()
        pred_right = pred_right.view(*shape) \
            .permute(0, 3, 1, 2).contiguous()

        
        gt_left = denormalize(gt_left)
        gt_right = denormalize(gt_right)
        
        pred_left = denormalize(pred_left)
        pred_right = denormalize(pred_right)
        pred_right.clamp_(0, 1)
        pred_left.clamp_(0, 1)
        
        res_left = metric_fn(pred_left, gt_left)
        res_right = metric_fn(pred_right, gt_right)
        res_avg = (res_left + res_right) / 2
        val_res_left.add(res_left.item(), inp_left.shape[0])
        val_res_right.add(res_right.item(), inp_right.shape[0])
        val_res_avg.add(res_avg.item(), inp_left.shape[0])
        time_avg.add(infer_time)

        if verbose:
            pbar.set_description('val_left {:.4f}  val_right {:.4f}  val_avg {:.4f}  infer_time {:.4f}'.format(val_res_left.item(), val_res_right.item(), val_res_avg.item(), time_avg.item()))
        
        
        if save_dir != None:
            save_imgs = {
                        f'{save_dir}/{filename}_L.png': pred_left.view(3, h_hr, w_hr).permute(1,2,0),
                        f'{save_dir}/{filename}_R.png': pred_right.view(3, h_hr, w_hr).permute(1,2,0),
                        f'{save_dir}/{filename}_disp.png': pred_disp.view(1, h_hr, w_hr).permute(1,2,0),
                    }
            for path, img in save_imgs.items():
                    img = img.cpu().numpy()
                    # img = (img * 255.0).round().astype(np.uint8)
                    if img.shape[-1] == 1:
                        img = img.round().astype(np.uint8).squeeze(-1)
                        img = Image.fromarray(img, mode='L')
                    else:
                        img = (img * 255.0).round().astype(np.uint8)
                        img = Image.fromarray(img)
                    img.save(path)

    return val_res_avg.item(), time_avg.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--scale_max', default='4')
    parser.add_argument('--fast', default=False)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    save_dir = os.path.join(config['save_dir'], args.model.split('/')[-2])
    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=8, pin_memory=True, collate_fn=dataset.collate_fn)

    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    res, infer_time = eval_psnr(loader, model,
        save_dir=save_dir,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        scale_max = int(args.scale_max),
        fast = args.fast,
        verbose=True)
    print('result: {:.4f}, time: {:.4f}'.format(res, infer_time))