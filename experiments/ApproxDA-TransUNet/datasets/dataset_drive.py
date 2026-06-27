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

    DRIVE layout (official public download — test GT not publicly available):
        {base_dir}/
          training/
            images/       — 21_training.tif ... 40_training.tif
            1st_manual/   — 21_manual1.gif  ... 40_manual1.gif

    Both train and test splits are drawn from the 20 labelled training images
    (80/20 split via generate_lists.py). All lookups go into training/.

    List files (one stem per line, without extension):
        {list_dir}/train.txt  — e.g.  21_training
        {list_dir}/test.txt   — e.g.  37_training  (held-out subset)

    test.py calls with split='test_vol'; remapped to 'test' transparently.
    """
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.base_dir = base_dir
        list_split = 'test' if split == 'test_vol' else split
        self.sample_list = [
            l for l in open(os.path.join(list_dir, list_split + '.txt')).readlines()
            if l.strip() and not l.strip().startswith('#')
        ]
        self.img_dir  = os.path.join(base_dir, 'training', 'images')
        self.mask_dir = os.path.join(base_dir, 'training', '1st_manual')

    def _load_image(self, stem):
        """Return grayscale float32 array in [0,1]."""
        for ext in ('.tif', '.tiff', '.png', '.jpg'):
            p = os.path.join(self.img_dir, stem + ext)
            if os.path.exists(p):
                return np.array(Image.open(p).convert('L'), dtype=np.float32) / 255.0
        raise FileNotFoundError(f"DRIVE: no image found for stem '{stem}' in {self.img_dir}")

    def _load_mask(self, stem):
        """Return binary uint8 array. Stem '21_training' → mask '21_manual1.gif'."""
        num_prefix = stem.split('_')[0]
        for ext in ('.gif', '.png', '.tif'):
            p = os.path.join(self.mask_dir, num_prefix + '_manual1' + ext)
            if os.path.exists(p):
                return (np.array(Image.open(p).convert('L')) > 127).astype(np.uint8)
        raise FileNotFoundError(f"DRIVE: no mask found for stem '{stem}' in {self.mask_dir}")

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
