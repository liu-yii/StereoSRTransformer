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
        _, _, disp, _=model.gen_feat(inp_left, inp_right)
        n = coord.shape[1]
        ql = 0
        preds_left = []
        preds_right = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred_left = model.query_rgb_left(coord[:, ql: qr, :], cell[:, ql: qr, :])
            pred_right = model.query_rgb_right(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds_left.append(pred_left)
            preds_right.append(pred_right)
            ql = qr
        pred_left = torch.cat(preds_left, dim=1)
        pred_right = torch.cat(preds_right, dim=1)
    return pred_left, pred_right


def eval_psnr(loader, model, save_dir, data_norm=None, eval_type=None, eval_bsize=None, window_size=0, scale_max=4, fast=False,
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
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    elif eval_type.startswith('AID'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='AID', scale=scale)
    else:
        raise NotImplementedError

    val_res_left = utils.Averager()
    val_res_right = utils.Averager()
    val_res_avg = utils.Averager()
    time_avg = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            if k != "filename":
                batch[k] = v.cuda()
        filename = batch['filename'][0].split('.')[0]
        inp_left, inp_right = torch.chunk((batch['inp'] - inp_sub) / inp_div, 2, dim=-1)
        gt_left, gt_right = torch.chunk(batch['gt'],2,dim=-1)
        # SwinIR Evaluation - reflection padding
        _, _, h_old, w_old = inp_left.size()
        if window_size != 0:
            for inp in [inp_left, inp_right]:
                h_pad = (h_old // window_size + 1) * window_size - h_old
                w_pad = (w_old // window_size + 1) * window_size - w_old
                inp = torch.cat([inp, torch.flip(inp, [2])], 2)[:, :, :h_old + h_pad, :]
                inp = torch.cat([inp, torch.flip(inp, [3])], 3)[:, :, :, :w_old + w_pad]
                
                coord = utils.make_coord((scale*(h_old+h_pad), scale*(w_old+w_pad))).unsqueeze(0).cuda()
                cell = torch.ones_like(coord)
                cell[:, :, 0] *= 2 / inp.shape[-2] / scale
                cell[:, :, 1] *= 2 / inp.shape[-1] / scale
        else:
            h_pad = 0
            w_pad = 0
            
            coord = batch['coord']
            cell = batch['cell']
        start_time = time.time()    
        if eval_bsize is None:
            with torch.no_grad():
                pred_left, pred_right, disp = model(inp_left, inp_right, coord, cell)
        else:
            if fast:
                pred_left, pred_right, disp = model(inp_left, inp_right, coord, cell*max(scale/scale_max, 1))
            else:
                pred_left, pred_right = batched_predict(model, inp_left, inp_right, coord, cell*max(scale/scale_max, 1), eval_bsize) # cell clip for extrapolation
        infer_time = time.time()-start_time
        
        pred_left = pred_left * gt_div + gt_sub
        pred_left.clamp_(0, 1)
        pred_right = pred_right * gt_div + gt_sub
        pred_right.clamp_(0, 1)
        for i, pred in enumerate([pred_left, pred_right]):   
            if save_dir != None:
                img = pred.clamp_(0, 1).view(scale*(h_old+h_pad), scale*(w_old+w_pad), 3).cpu().numpy()
                img = (img * 255.0).round().astype(np.uint8)
                img = Image.fromarray(img)
                if i == 0:
                    img.save(f'{save_dir}/{filename}_L.png')
                else:
                    img.save(f'{save_dir}/{filename}_R.png')
                           
        if eval_type is not None and fast == False: # reshape for shaving-eval
            # gt reshape
            ih, iw = inp_left.shape[-2:]
            iw = iw
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [inp_left.shape[0], round(ih * s), round(iw * s), 3]
            gt_left = gt_left.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            gt_right = gt_right.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            
            # prediction reshape
            ih += h_pad
            iw += w_pad
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
        res_avg = (res_left + res_right)/2
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
    parser.add_argument('--window', default='0')
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

    res,infer_time = eval_psnr(loader, model,
        save_dir=save_dir,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        window_size=int(args.window),
        scale_max = int(args.scale_max),
        fast = args.fast,
        verbose=True)
    print('result: {:.4f}, time: {:.4f}'.format(res, infer_time))