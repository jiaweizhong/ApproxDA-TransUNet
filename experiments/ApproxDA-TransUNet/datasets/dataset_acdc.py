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
    def __init__(self, output_size, hr_size=None):
        self.output_size = output_size
        # hr_size (--point_loss): also return the augmented label on a fixed hr_size canvas (nearest), the
        # native-resolution target of the point supervision loss
        self.hr_size = hr_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if self.hr_size:
            hx, hy = (self.hr_size, self.hr_size) if isinstance(self.hr_size, int) else self.hr_size
            label_hr = label if (x, y) == (hx, hy) else zoom(label, (hx / x, hy / y), order=0)
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(
                image, (self.output_size[0] / x, self.output_size[1] / y), order=3
            )
            label = zoom(
                label, (self.output_size[0] / x, self.output_size[1] / y), order=0
            )
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {"image": image, "label": label.long()}
        if self.hr_size:
            sample["label_hr"] = torch.from_numpy(np.ascontiguousarray(label_hr)).long()
        return sample


class ACDC_dataset(Dataset):
    """
    ACDC cardiac cine MRI, MT-UNet preprocessed release: the 70 / 10 / 20 patient split used by TransUNet, Swin-UNet,
    MT-UNet and others. base_dir = .../ACDC_MTUNet with train/, valid/ and test/ sub-directories:
      split 'train'    -> train/{name}: 2-D slice, keys 'img' (224 x 224) and 'label'
      split 'val'      -> valid/{name}: 2-D slice
      split 'test_vol' -> test/{name}:  volume (D x 224 x 224), one per patient and phase (ED / ES)
    Intensities are already normalised to [-1, 1]. Labels: 0 background, 1 RV, 2 Myo, 3 LV (num_classes = 4).
    """

    SUBDIR = {"train": "train", "val": "valid", "test_vol": "test"}

    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        with open(os.path.join(list_dir, split + ".txt")) as f:
            self.sample_list = [line.strip() for line in f if line.strip()]
        self.data_dir = os.path.join(base_dir, self.SUBDIR[split])

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        name = self.sample_list[idx]
        data = np.load(os.path.join(self.data_dir, name))
        sample = {"image": data["img"].astype(np.float32), "label": data["label"].astype(np.float32)}
        if self.transform:
            sample = self.transform(sample)
        sample["case_name"] = name[: -len(".npz")] if name.endswith(".npz") else name
        return sample
