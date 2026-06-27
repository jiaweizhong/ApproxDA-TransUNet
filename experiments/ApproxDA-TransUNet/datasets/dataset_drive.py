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


class DRIVE_dataset(Dataset):
    """
    DRIVE retinal vessel segmentation dataset (binary).

    Standard DRIVE layout (official download):
        {base_dir}/
          training/
            images/       — 21_training.tif ... 40_training.tif
            1st_manual/   — 21_manual1.gif  ... 40_manual1.gif
          test/
            images/       — 01_test.tif ... 20_test.tif
            1st_manual/   — 01_manual1.gif ... 20_manual1.gif

    Also supports a flat preprocessed layout:
        {base_dir}/
          images/   — *.png or *.tif
          masks/    — *.png (binary 0/255)

    List files (one stem per line, without extension):
        {list_dir}/train.txt  — e.g.  21_training
        {list_dir}/test.txt   — e.g.  01_test

    test.py calls with split='test_vol'; remapped to 'test' transparently.
    """
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.base_dir = base_dir
        list_split = 'test' if split == 'test_vol' else split
        self.split = list_split
        self.sample_list = [
            l for l in open(os.path.join(list_dir, list_split + '.txt')).readlines()
            if l.strip() and not l.strip().startswith('#')
        ]
        # Auto-detect flat vs hierarchical layout
        self._flat = os.path.isdir(os.path.join(base_dir, 'images'))

    def _load_image(self, stem):
        """Try common image extensions; return grayscale float32 array in [0,1]."""
        if self._flat:
            dirs = [os.path.join(self.base_dir, 'images')]
        else:
            sub = 'training' if self.split == 'train' else 'test'
            dirs = [os.path.join(self.base_dir, sub, 'images')]
        for d in dirs:
            for ext in ('.tif', '.tiff', '.png', '.jpg'):
                p = os.path.join(d, stem + ext)
                if os.path.exists(p):
                    return np.array(Image.open(p).convert('L'), dtype=np.float32) / 255.0
        raise FileNotFoundError(f"DRIVE: no image found for stem '{stem}' in {dirs}")

    def _load_mask(self, stem):
        """Try standard DRIVE mask naming conventions; return binary uint8 array."""
        if self._flat:
            mask_dir = os.path.join(self.base_dir, 'masks')
            for ext in ('.png', '.gif', '.tif'):
                p = os.path.join(mask_dir, stem + ext)
                if os.path.exists(p):
                    return (np.array(Image.open(p).convert('L')) > 127).astype(np.uint8)
        else:
            sub = 'training' if self.split == 'train' else 'test'
            mask_dir = os.path.join(self.base_dir, sub, '1st_manual')
            # stem = '21_training' → number prefix '21'
            num_prefix = stem.split('_')[0]
            for ext in ('.gif', '.png', '.tif'):
                p = os.path.join(mask_dir, num_prefix + '_manual1' + ext)
                if os.path.exists(p):
                    return (np.array(Image.open(p).convert('L')) > 127).astype(np.uint8)
        raise FileNotFoundError(f"DRIVE: no mask found for stem '{stem}'")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        name = self.sample_list[idx].strip('\n')
        image = self._load_image(name)
        label = self._load_mask(name)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = name
        return sample
