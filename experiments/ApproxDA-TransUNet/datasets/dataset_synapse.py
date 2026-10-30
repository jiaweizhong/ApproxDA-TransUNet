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


def random_rot(image, label):
    """90-degree rotations only (--aug rot_only): a proper rotation keeps left/right handedness, no mirror image."""
    k = np.random.randint(0, 4)
    return np.rot90(image, k).copy(), np.rot90(label, k).copy()


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    """aug: 'default' = TransUNet (p 0.5 rot90 + flip, which mirrors half the slices without swapping the
    left/right kidney labels; else p 0.25 rotation in [-20, 20) degrees); 'rot_only' = same but rot90 without
    the flip; 'small_rot' = only the small rotation, p 0.5 (orientation and handedness kept)."""

    def __init__(self, output_size, aug="default", hr_size=None):
        self.output_size = output_size
        self.aug = aug
        # hr_size (--point_refine): also return the augmented slice and label resized to hr_size x hr_size
        # (image linear, label nearest) as the native-resolution training canvas of the point head
        self.hr_size = hr_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        if self.aug == "small_rot":
            if random.random() > 0.5:
                image, label = random_rotate(image, label)
        elif random.random() > 0.5:
            if self.aug == "rot_only":
                image, label = random_rot(image, label)
            else:
                image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if self.hr_size:
            f = (self.hr_size / x, self.hr_size / y)
            image_hr = image if f == (1.0, 1.0) else zoom(image, f, order=1)
            label_hr = label if f == (1.0, 1.0) else zoom(label, f, order=0)
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(
                image, (self.output_size[0] / x, self.output_size[1] / y), order=3
            )  # why not 3?
            label = zoom(
                label, (self.output_size[0] / x, self.output_size[1] / y), order=0
            )
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {"image": image, "label": label.long()}
        if self.hr_size:
            sample["image_hr"] = torch.from_numpy(np.ascontiguousarray(image_hr, dtype=np.float32))
            sample["label_hr"] = torch.from_numpy(np.ascontiguousarray(label_hr)).long()
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + ".txt")).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip("\n")
            data_path = os.path.join(self.data_dir, slice_name + ".npz")
            data = np.load(data_path)
            image, label = data["image"], data["label"]
        else:
            vol_name = self.sample_list[idx].strip("\n")
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data["image"][:], data["label"][:]

        sample = {"image": image, "label": label}
        if self.transform:
            sample = self.transform(sample)
        sample["case_name"] = self.sample_list[idx].strip("\n")
        return sample
