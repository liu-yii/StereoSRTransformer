import os
import json
from PIL import Image

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob
from utils import readPFM
from datasets import register

@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == 'none':
                self.files.append({
                        'img': transforms.ToTensor()(Image.open(file).convert('RGB')),
                        'filename': filename
                    })

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(imageio.imread(file), f)
                    print('dump', bin_file)
                self.files.append( {
                        'img': transforms.ToTensor()(Image.open(file).convert('RGB')),
                        'filename': filename
                    })

            elif cache == 'in_memory':
                self.files.append({
                        'img': transforms.ToTensor()(Image.open(file).convert('RGB')),
                        'filename': filename
                    })

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            return x

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            return x


@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx]['img'], self.dataset_2[idx]['img'], self.dataset_1[idx]['filename']
    
@register('stereo-image-folders')
class StereoImageFolders(Dataset):
    def __init__(self, root_path, split_file=None, split_key=None, first_k=None, phase='train',
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache
        self.repeat = repeat
        self.cache = cache
        
        # if phase == 'train':
        #     filenames = sorted(os.listdir(root_path))
        filenames = os.listdir(root_path)
        if first_k is not None:
            filenames = filenames[:first_k]
        
        self.files = []
        for filename in filenames:
            file_l = os.path.join(root_path, filename + '/hr0.png')
            file_r = os.path.join(root_path, filename + '/hr1.png')
            file_disp = os.path.join(root_path, filename + '/disp_occ.png')

            if cache == 'none':
                self.files.append({
                        'img_l': transforms.ToTensor()(Image.open(file_l).convert('RGB')),
                        'img_r': transforms.ToTensor()(Image.open(file_r).convert('RGB')),
                        'disp': transforms.ToTensor()(np.array(Image.open(file_disp)).astype(np.float32) / 256.),
                        'filename': filename
                    })

            elif cache == 'in_memory':
                self.files.append({
                        'img_l': transforms.ToTensor()(Image.open(file_l).convert('RGB')),
                        'img_r': transforms.ToTensor()(Image.open(file_r).convert('RGB')),
                        'disp': transforms.ToTensor()(np.array(Image.open(file_disp)).astype(np.float32) / 256.),
                        'filename': filename
                    })

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            return x

        elif self.cache == 'in_memory':
            return x
        

@register('stereo-image-folders-without-disp')
class StereoImageFoldersTest(Dataset):
    def __init__(self, root_path, split_file=None, split_key=None, first_k=None, phase='train',
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache
        self.repeat = repeat
        self.cache = cache

        # if phase == 'train':
        #     filenames = sorted(os.listdir(root_path))
        filenames = os.listdir(root_path)
        if first_k is not None:
            filenames = filenames[:first_k]
        
        self.files = []
        for filename in filenames:
            file_l = os.path.join(root_path, filename + '/hr0.png')
            file_r = os.path.join(root_path, filename + '/hr1.png')
            # file_disp = os.path.join(root_path, filename + '/disp_occ.png')

            if cache == 'none':
                self.files.append({
                        'img_l': transforms.ToTensor()(Image.open(file_l).convert('RGB')),
                        'img_r': transforms.ToTensor()(Image.open(file_r).convert('RGB')),
                        'filename': filename
                    })

            elif cache == 'in_memory':
                self.files.append({
                        'img_l': transforms.ToTensor()(Image.open(file_l).convert('RGB')),
                        'img_r': transforms.ToTensor()(Image.open(file_r).convert('RGB')),
                        'filename': filename
                    })

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            return x

        elif self.cache == 'in_memory':
            return x


@register('stereo-image-folders-sceneflow')
class StereoImageFolders(Dataset):
    def __init__(self, root_path, split_file=None, split_key=None, first_k=None, phase='TRAIN',
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache
        self.repeat = repeat
        self.cache = cache
        
        split = 'frames_cleanpass'
        left_files = sorted(glob(root_path + '/' + split + '/' + phase + '/*/*/left/*.png'))
        self.samples = []
        for left_name in left_files:
            sample = dict()
            sample['left'] = left_name
            sample['right'] = left_name.replace('/left/', '/right/')
            sample['disp'] = left_name.replace(split, 'disparity')[:-4] + '.pfm'
            sample['filename'] = left_name.split('/')[-1]
            self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples
        file_l = sample[idx]['left']
        file_r = sample[idx]['right']
        file_disp = sample[idx]['disp']
        filename = sample[idx]['filename']
        x ={
                'img_l': transforms.ToTensor()(Image.open(file_l).convert('RGB')),
                'img_r': transforms.ToTensor()(Image.open(file_r).convert('RGB')),
                'disp': transforms.ToTensor()(np.array(readPFM(file_disp)[0])),
                'filename': filename
            }

        if self.cache == 'none':
            return x

        elif self.cache == 'in_memory':
            return x
        