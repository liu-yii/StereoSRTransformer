import argparse
import os
import random
import numpy as np

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR

import datasets
import models
import utils
from utils import warp, loss_disp_smoothness, warp_coord, denormalize
from test import eval_psnr
# torch.backends.cudnn.enabled = False


def mixup(lq, gt, alpha=1.2):
    if random.random() < 0.5:
        return lq, gt

    v = np.random.beta(alpha, alpha)
    r_index = torch.randperm(lq.size(0)).to(gt.device)

    lq = v * lq + (1 - v) * lq[r_index, :]
    gt = v * gt + (1 - v) * gt[r_index, :]
    return lq, gt

def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    
    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0][0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=(tag == 'train' or tag == 'val'), num_workers=0, pin_memory=True, collate_fn=dataset.collate_fn)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def prepare_training():
#     if config.get('resume') is not None:
    if config.get('resume') is not None and os.path.exists(config.get('resume')):
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        if config.get('cosine_annealing'):
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=config['epoch_max'], eta_min=config['cosine_annealing']['eta_min'])
        elif config.get('multi_step_lr'):
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        else:
            lr_scheduler = None
        # for _ in range(epoch_start - 1):
        #     lr_scheduler.step()
        log('Resume training, epoch: #{}'.format(sv_file['epoch']))
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('cosine_annealing'):
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=config['epoch_max'], eta_min=config['cosine_annealing']['eta_min'])
        elif config.get('multi_step_lr'):
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        else:
            lr_scheduler = None
            
    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model, optimizer, epoch, finetune = False):
    model.train()
    loss_fn = nn.L1Loss()
    train_loss = utils.Averager()
    train_loss_rgb = utils.Averager()
    train_loss_disp = utils.Averager()
    # metric_fn = utils.calc_psnr
    
    for idx, batch in enumerate(tqdm(train_loader, leave=False, desc='train')):
        data, filename = batch
        for k, v in data.items():
            if isinstance(data[k], list):
                for idx in range(len(data[k])):
                    data[k][idx] = v[idx].cuda()
            else:
                if v is not None:
                    data[k] = v.cuda()
        inp = data['inp']
        bs = len(inp)
        scale = torch.Tensor([data['gt'][i].shape[-1] / inp[i].shape[-1] for i in range(bs)]).cuda()
        inp = torch.stack(inp, dim=0)
        coord = torch.stack(data['coord'], dim=0)
        cell = torch.stack(data['cell'], dim=0)
        # if config["phase"] == "train" and config["use_mixup"]:
        #     inp, gt = mixup(inp, gt)
        inp_left, inp_right = torch.chunk(inp, 2, dim=1)
        
        
        pred_left, pred_right, pred_disp = model(inp_left, inp_right, coord, cell, scale)
        losses = 0.0
        for i in range(bs):
            if i == bs - 1:
                gt = data['gt'][i].unsqueeze(0)
                coord = data['coord'][i].unsqueeze(0)
                
                # GT color
                gt_left, gt_right = torch.chunk(gt, 2, dim=1)
                gt_l_rgb = F.grid_sample(gt_left, coord.flip(-1).unsqueeze(1), mode='nearest', padding_mode='border')[:, :, 0, :] \
                            .permute(0, 2, 1)
                gt_r_rgb = F.grid_sample(gt_right, coord.flip(-1).unsqueeze(1), mode='nearest', padding_mode='border')[:, :, 0, :] \
                            .permute(0, 2, 1)
                
                # GT disparity
                if not finetune:
                    disp = data['disp'][i].unsqueeze(0)
                    gt_disp = F.grid_sample(disp, coord.flip(-1).unsqueeze(1), mode='nearest', padding_mode='border')[:, :, 0, :] \
                            .permute(0, 2, 1)
                else:
                    gt_disp = None
                loss_rgb = loss_fn(pred_left[i].unsqueeze(0), gt_l_rgb) + loss_fn(pred_right[i].unsqueeze(0), gt_r_rgb)
                warp_left = warp_coord(coord, pred_disp[i].unsqueeze(0), gt_right)
                loss_unsupervised = loss_fn(warp_left, gt_l_rgb)
                lambda_loss = 0.1
                if not finetune:
                    loss_disp = loss_fn(pred_disp[i].unsqueeze(0), gt_disp) + loss_unsupervised
                else:
                    loss_disp = loss_unsupervised
                loss = loss_rgb + lambda_loss * loss_disp
                # psnr = metric_fn(pred, gt)
                train_loss.add(loss.item())
                train_loss_rgb.add(loss_rgb.item())
                train_loss_disp.add(loss_disp.item())
                losses += loss
        

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        pred_left = None
        pred_right = None
        losses = None
        
    return train_loss.item(), train_loss_rgb.item(), train_loss_disp.item()


def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss, train_loss_rgb, train_loss_disp = train(train_loader, model, optimizer, \
                           epoch, finetune = config.get('finetune'))
        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('train: loss_total={:.4f} loss_rgb={:.4f} loss_disp={:.4f}'.\
                        format(train_loss, train_loss_rgb, train_loss_disp))
        writer.add_scalars('loss', {'train': train_loss, 'rgb': train_loss_rgb, 'disp': train_loss_disp}, epoch)

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            if n_gpus > 1 and (config.get('eval_bsize') is not None):
                model_ = model.module
            else:
                model_ = model
            val_res, _ = eval_psnr(val_loader, model_,
                save_dir= None,
                data_norm= None,
                eval_type=config.get('eval_type'),
                eval_bsize=config.get('eval_bsize'),
                verbose=True)

            log_info.append('val: psnr={:.4f}'.format(val_res))
            writer.add_scalars('psnr', {'val': val_res}, epoch)

            # for path, img in save_imgs:
            #     writer.add_image(path, img, epoch, dataformats='HWC')

            if val_res > max_val_v:
                max_val_v = val_res
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_dir = config['save_dir']
    save_name = args.name
    if save_name is None:
        save_name = os.path.split(args.config)[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join(save_dir, save_name)
    
    main(config, save_path)