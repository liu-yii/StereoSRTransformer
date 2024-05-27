import argparse
import os
import math
from functools import partial
import matplotlib.pyplot as plt

import yaml
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils
from utils import make_coord
import cv2
import numpy as np
from PIL import Image
import lpips
import time



def batched_predict(model, inp_left, inp_right, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp_left, inp_right)
        n = coord.shape[1]
        ql = 0
        preds_left = []
        preds_right = []
        disps_l2r = []
        disps_r2l = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred_left, disp_r2l, _ = model.query_rgb_left(coord[:, ql: qr, :], cell[:, ql: qr, :])
            pred_right, disp_l2r, _ = model.query_rgb_right(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds_left.append(pred_left)
            preds_right.append(pred_right)
            disps_l2r.append(disp_l2r)
            disps_r2l.append(disp_r2l)
            ql = qr
        pred_left = torch.cat(preds_left, dim=1)
        pred_right = torch.cat(preds_right, dim=1)
        disp_l2r = torch.cat(disps_l2r, dim=1)
        disp_r2l = torch.cat(disps_r2l, dim=1)
    return (pred_left, disp_r2l), (pred_right, disp_l2r)


def eval_psnr(loader, model, save_dir, data_norm=None, eval_type=None, eval_bsize=None, scale_max=4, fast=False,
              verbose=False):
    if save_dir != None:
        os.makedirs(save_dir, exist_ok=True)
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

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
        data, filename, raw_hr = batch
        for k, v in data.items():
            data[k] = v.cuda()
        filename = filename[0].split('.')[0]
        inp_left, inp_right = torch.chunk((data['inp'] - inp_sub) / inp_div, 2, dim=-1)
        gt_left, gt_right = torch.chunk(data['gt'], 2, dim=-1)
        # SwinIR Evaluation - reflection padding
        _, _, h, w = inp_left.size()
        coord = data['coord']
        cell = data['cell']
        start_time = time.time()    
        if eval_bsize is None:
            with torch.no_grad():
                preds_left, preds_right, _, _ = model(inp_left, inp_right, coord, cell)
        else:
            if fast:
                preds_left, preds_right, _, _ = model(inp_left, inp_right, coord, cell*max(scale/scale_max, 1))
            else:
                preds_left, preds_right = batched_predict(model, inp_left, inp_right, coord, cell*max(scale/scale_max, 1), eval_bsize) # cell clip for extrapolation
        infer_time = time.time()-start_time
        
        pred_left, pred_right = preds_left[0], preds_right[0]
        disp_l2r, disp_r2l = preds_left[1], preds_right[1]
        pred_left = pred_left * gt_div + gt_sub
        pred_left.clamp_(0, 1)
        pred_right = pred_right * gt_div + gt_sub
        pred_right.clamp_(0, 1)
        
        # for i, pred in enumerate([pred_left, pred_right]):   
        #     if save_dir != None:
        #         img = pred.view(scale*h, scale*w, 3).cpu().numpy()
        #         img = (img * 255.0).round().astype(np.uint8)
        #         img = Image.fromarray(img)
        #         if i == 0:
        #             img.save(f'{save_dir}/{filename}_L.png')
        #         elif i == 1:
        #             img.save(f'{save_dir}/{filename}_R.png')    
        # if save_dir != None:
        #     img = disp_l2r.view(scale*h, scale*w, 1).cpu().numpy()
        #     img = (img).round().astype(np.uint8).squeeze(-1)
        #     img = Image.fromarray(img,mode='L')
        #     img.save(f'{save_dir}/{filename}_disp.png')
        if save_dir != None:
            save_imgs = {
                        f'{save_dir}/{filename}_L.png': pred_left.view(scale*h, scale*w, 3),
                        f'{save_dir}/{filename}_R.png': pred_right.view(scale*h, scale*w, 3),
                        f'{save_dir}/{filename}_disp.png': disp_l2r.view(scale*h, scale*w, 1)
                    }
            for path, img in save_imgs.items():
                    img = img.cpu().numpy()
                    # img = (img * 255.0).round().astype(np.uint8)
                    if img.shape[-1] == 1:
                        img = (img*scale*w/2.).round().astype(np.uint8).squeeze(-1)
                        img = Image.fromarray(img, mode='L')
                    else:
                        img = (img * 255.0).round().astype(np.uint8)
                        img = Image.fromarray(img)
                    img.save(path)
        
        if eval_type is not None and fast == False: # reshape for shaving-eval
            # gt reshape
            ih, iw = inp_left.shape[-2:]
            iw = iw
            s = math.sqrt(data['coord'].shape[1] / (ih * iw))
            shape = [inp_left.shape[0], round(ih * s), round(iw * s), 3]
            gt_left = gt_left.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            gt_right = gt_right.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            
            # prediction reshape
            s = math.sqrt(coord.shape[1] / (ih * iw))
            shape = [inp_left.shape[0], round(ih * s), round(iw * s), 3]
            pred_left = pred_left.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            pred_left = pred_left[..., :gt_left.shape[-2], :gt_left.shape[-1]]
            pred_right = pred_right.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            pred_right = pred_right[..., :gt_right.shape[-2], :gt_right.shape[-1]]
            
        res_left = metric_fn(pred_left, gt_left)
        res_right = metric_fn(pred_right, gt_right)
        res_avg = (res_left + res_right) / 2
        val_res_left.add(res_left.item(), inp_left.shape[0])
        val_res_right.add(res_right.item(), inp_right.shape[0])
        val_res_avg.add(res_avg.item(), inp_left.shape[0])
        time_avg.add(infer_time)

        if verbose:
            pbar.set_description('val_left {:.4f}  val_right {:.4f}  val_avg {:.4f}  infer_time {:.4f}'.format(val_res_left.item(), val_res_right.item(), val_res_avg.item(), time_avg.item()))
            
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
        num_workers=8, pin_memory=True)

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