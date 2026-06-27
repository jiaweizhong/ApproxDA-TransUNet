import os
import random
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
import torch
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=3, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        return {'image': image, 'label': label.long()}


def _find_dirs(base_dir):
    """Auto-detect CVC-ClinicDB directory layout.

    Supports two common layouts:
      A) Kaggle default:  PNG/Original/  +  PNG/Ground Truth/
      B) Flat standard:   images/        +  masks/
    """
    kaggle_img  = os.path.join(base_dir, 'PNG', 'Original')
    kaggle_mask = os.path.join(base_dir, 'PNG', 'Ground Truth')
    if os.path.isdir(kaggle_img):
        return kaggle_img, kaggle_mask
    return os.path.join(base_dir, 'images'), os.path.join(base_dir, 'masks')


def _open_img(directory, stem):
    for ext in ('.png', '.jpg', '.jpeg', '.tif'):
        p = os.path.join(directory, stem + ext)
        if os.path.exists(p):
            return Image.open(p)
    raise FileNotFoundError(f"CVC: no file for stem '{stem}' in {directory}")


class CVC_dataset(Dataset):
    """
    CVC-ClinicDB colonoscopy polyp segmentation dataset (binary).
    612 images from 29 colonoscopy sequences.

    Supports two directory layouts (auto-detected):
      Kaggle:    {base_dir}/PNG/Original/    + {base_dir}/PNG/Ground Truth/
      Standard:  {base_dir}/images/          + {base_dir}/masks/

    List files (one stem per line, no extension):
        {list_dir}/train.txt  — e.g. 1, 2, ... 490
        {list_dir}/test.txt   — e.g. 491, ... 612

    test.py calls with split='test_vol'; remapped to 'test' transparently.
    """
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        list_split = 'test' if split == 'test_vol' else split
        self.sample_list = [
            l for l in open(os.path.join(list_dir, list_split + '.txt')).readlines()
            if l.strip() and not l.strip().startswith('#')
        ]
        self.img_dir, self.mask_dir = _find_dirs(base_dir)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        name  = self.sample_list[idx].strip('\n')
        image = np.array(_open_img(self.img_dir,  name).convert('L'), dtype=np.float32) / 255.0
        label = (np.array(_open_img(self.mask_dir, name).convert('L')) > 127).astype(np.uint8)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = name
        return sample
