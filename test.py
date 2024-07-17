import argparse
import os
import math
from functools import partial

import yaml
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils
from PIL import Image
import numpy as np


def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        feat_left, feat_right, corr = model.gen_feat(inp)
        raw_disp, refined_left, refined_right = model.refine_corr(feat_left, feat_right, corr)
        inp_l, inp_r = inp.chunk(2, dim=1)
        n = coord.shape[1]
        ql = 0
        preds_l = []
        preds_r = []
        disps = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred_r,_ = model.query_rgb(inp_r, refined_right, coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds_r.append(pred_r)
            pred_l, disp = model.query_rgb(inp_l, refined_left, coord[:, ql: qr, :], cell[:, ql: qr, :], raw_disp)
            preds_l.append(pred_l)
            disps.append(disp)
            del pred_r, pred_l, disp
            ql = qr
        pred_l = torch.cat(preds_l, dim=1)
        pred_r = torch.cat(preds_r, dim=1)
        pred_disp = torch.cat(disps, dim=1)
        pred = torch.cat([pred_l, pred_r], dim=-1)
    return pred, pred_disp


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None, window_size=0, scale_max=4, fast=False, save_dir=None,
              verbose=False):
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
    else:
        raise NotImplementedError

    val_res = utils.Averager()
    save_idx = 1
    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        # SwinIR Evaluation - reflection padding
        if window_size != 0:
            _, _, h_old, w_old = inp.size()
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
            _, _, h_old, w_old = inp.size()
            coord = batch['coord']
            cell = batch['cell']
            
        if eval_bsize is None:
            with torch.no_grad():
                out = model(inp, coord, cell)
                pred = out['out_rgb']
                pred_disp = out['disp']
        else:
            if fast:
                with torch.no_grad():
                    out = model(inp, coord, cell*max(scale/scale_max, 1))
                    pred = out['out_rgb']
                    pred_disp = out['disp']
            else:
                pred, pred_disp = batched_predict(model, inp, coord, cell*max(scale/scale_max, 1), eval_bsize) # cell clip for extrapolation
            
        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)
        pred_l, pred_r = pred.chunk(2, dim=-1)
        gt_left, gt_right = batch['gt'].chunk(2, dim=-1)
        if eval_type is not None: 
            if save_dir is not None:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                # img_l
                img_l = pred_l.clamp_(0, 1).view(scale*(h_old+h_pad), scale*(w_old+w_pad), 3).cpu().numpy()
                img_l = (img_l * 255.0).round().astype(np.uint8)
                img_l = Image.fromarray(img_l)
                img_l.save(save_dir + '/{:0>4d}_l.png'.format(save_idx))
                # img_r
                img_r = pred_r.clamp_(0, 1).view(scale*(h_old+h_pad), scale*(w_old+w_pad), 3).cpu().numpy()
                img_r = (img_r * 255.0).round().astype(np.uint8)
                img_r = Image.fromarray(img_r)
                img_r.save(save_dir + '/{:0>4d}_r.png'.format(save_idx))

                # img_disp
                img_disp = pred_disp.clamp_(0).view(scale*(h_old+h_pad), scale*(w_old+w_pad), 1).cpu().numpy()
                img_disp = img_disp.round().astype(np.uint8)
                img_disp = Image.fromarray(img_disp[:,:,0], mode='L')
                img_disp.save(save_dir + '/{:0>4d}_disp.png'.format(save_idx))
                save_idx += 1
        if eval_type is not None and fast == False: # reshape for shaving-eval
            # gt reshape
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            gt_left = gt_left.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            gt_right = gt_right.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            
            # prediction reshape
            ih += h_pad
            iw += w_pad
            s = math.sqrt(coord.shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred_l = pred_l.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            pred_l = pred_l[..., :gt_left.shape[-2], :gt_left.shape[-1]]

            pred_r = pred_r.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            pred_r = pred_r[..., :gt_right.shape[-2], :gt_right.shape[-1]]
            
        res = (metric_fn(pred_l, gt_left) + metric_fn(pred_r, gt_right))/2
        val_res.add(res.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))
            
    return val_res.item()


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

    save_path = args.model.split('/')[-2] + "_" + args.config.split('-')[-1][:-len('.yaml')] + "xSR"
    save_dir = os.path.join(config['save_dir'], save_path)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=8, pin_memory=True)

    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    res = eval_psnr(loader, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        window_size=int(args.window),
        scale_max = int(args.scale_max),
        fast = args.fast,
        save_dir = save_dir,
        verbose=True)
    print('result: {:.4f}'.format(res))