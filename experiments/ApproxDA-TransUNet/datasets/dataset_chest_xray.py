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


def _load_image(img_dir, stem):
    for ext in ('.png', '.jpg', '.jpeg', '.PNG', '.JPG'):
        p = os.path.join(img_dir, stem + ext)
        if os.path.exists(p):
            return np.array(Image.open(p).convert('L'), dtype=np.float32) / 255.0
    raise FileNotFoundError(f'ChestXray: no image for stem {stem!r} in {img_dir}')


def _load_mask(base_dir, stem):
    """
    Support two common Montgomery County layouts:

    Layout A (preprocessed, combined mask):
        masks/<stem>.png

    Layout B (original MontgomerySet format):
        ManualMask/leftMask/<stem>_mask.png
        ManualMask/rightMask/<stem>_mask.png
        → merged via logical OR

    Falls back A→B in order.
    """
    # Layout A: flat masks/ directory
    mask_dir = os.path.join(base_dir, 'masks')
    for ext in ('.png', '.jpg', '.PNG'):
        p = os.path.join(mask_dir, stem + ext)
        if os.path.exists(p):
            return (np.array(Image.open(p).convert('L')) > 127).astype(np.uint8)

    # Layout B: ManualMask/leftMask + rightMask
    left_dir  = os.path.join(base_dir, 'ManualMask', 'leftMask')
    right_dir = os.path.join(base_dir, 'ManualMask', 'rightMask')
    left_mask = right_mask = None
    for ext in ('.png', '.PNG', '.jpg'):
        lp = os.path.join(left_dir,  stem + '_mask' + ext)
        rp = os.path.join(right_dir, stem + '_mask' + ext)
        if os.path.exists(lp):
            left_mask = (np.array(Image.open(lp).convert('L')) > 127)
        if os.path.exists(rp):
            right_mask = (np.array(Image.open(rp).convert('L')) > 127)
        if left_mask is not None or right_mask is not None:
            break
    if left_mask is None and right_mask is None:
        raise FileNotFoundError(f'ChestXray: no mask found for stem {stem!r} under {base_dir}')
    if left_mask is None:
        return right_mask.astype(np.uint8)
    if right_mask is None:
        return left_mask.astype(np.uint8)
    return (left_mask | right_mask).astype(np.uint8)


class ChestXray_dataset(Dataset):
    """
    Chest X-ray lung segmentation dataset (binary, Montgomery County format).

    Supports two directory layouts (see _load_mask above):
      Layout A — flat masks/:
        {base_dir}/images/<stem>.png
        {base_dir}/masks/<stem>.png

      Layout B — original MontgomerySet:
        {base_dir}/CXR_png/<stem>.png          (or images/)
        {base_dir}/ManualMask/leftMask/<stem>_mask.png
        {base_dir}/ManualMask/rightMask/<stem>_mask.png

    List files (one stem per line, no extension):
        {list_dir}/train.txt
        {list_dir}/test.txt

    138 images total; default 80/20 split via generate_lists.py.
    test.py calls with split='test_vol'; remapped to 'test' transparently.
    """
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.base_dir  = base_dir
        list_split = 'test' if split == 'test_vol' else split
        self.sample_list = [
            l for l in open(os.path.join(list_dir, list_split + '.txt')).readlines()
            if l.strip() and not l.strip().startswith('#')
        ]
        # Support both Layout A (images/) and Layout B (CXR_png/)
        img_dir_a = os.path.join(base_dir, 'images')
        img_dir_b = os.path.join(base_dir, 'CXR_png')
        self.img_dir = img_dir_a if os.path.isdir(img_dir_a) else img_dir_b

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        name  = self.sample_list[idx].strip('\n')
        image = _load_image(self.img_dir, name)
        label = _load_mask(self.base_dir, name)

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = name
        return sample
