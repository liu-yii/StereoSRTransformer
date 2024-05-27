from PIL import Image
import os
import tqdm

Datasets = ['Flickr1024']

path = '/media/yi/E/Research/Datasets/Flickr1024/Validation'
save_dir = '/media/yi/E/Research/Datasets/iPASSR/val/processed'

idx = 0
for dataset in Datasets:
    # dir = os.path.join(path, dataset)
    dir = path
    files = os.listdir(dir)
    length = len(files)
    for i in tqdm.tqdm(range(length//2)):
        file_l = os.path.join(dir, '{:03}_L.png'.format(i+1))
        file_r = os.path.join(dir, '{:03}_R.png'.format(i+1))
        img_l = Image.open(file_l).convert('RGB')
        img_r = Image.open(file_r).convert('RGB')
        save_path = os.path.join(save_dir, '{:04}'.format(idx+1))
        if os.path.exists(save_path) is False:
            os.mkdir(save_path)
        img_l.save(os.path.join(save_path, 'hr_0.png'))
        img_r.save(os.path.join(save_path, 'hr_1.png'))
        idx += 1

