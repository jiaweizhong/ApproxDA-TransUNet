import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
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
    image = ndimage.rotate(image, angle, order=0, reshape=False)
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


def _normalize(image):
    lo, hi = image.min(), image.max()
    if hi - lo < 1e-8:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - lo) / (hi - lo)).astype(np.float32)


class ACDC_dataset(Dataset):
    """
    ACDC cardiac MRI dataset (Kaggle preprocessed format).

    Training split:
      base_dir = .../ACDC_training_slices/
      Reads {slice_name}.h5 with keys 'image' (H×W float32), 'label' (H×W uint8)

    Test split (split='test_vol'):
      base_dir = .../ACDC_training_volumes/
      Reads {vol_name}.h5 with keys 'image' (D×H×W float32), 'label' (D×H×W uint8)

    Labels: 0=background, 1=RV, 2=Myo, 3=LV  →  num_classes=4
    """
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        name = self.sample_list[idx].strip('\n')
        filepath = os.path.join(self.data_dir, name + '.h5')

        with h5py.File(filepath, 'r') as f:
            image = f['image'][:]
            label = f['label'][:]

        image = _normalize(image.astype(np.float32))
        label = label.astype(np.uint8)

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = name
        return sample
