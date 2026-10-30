"""Synapse-format slices with their adjacent slices, for --slice_attn K: a training sample is the stack of the 2K+1
slices z-K..z+K of one volume (indices clamped at the volume ends) with the label of the centre slice; every slice of
the stack gets the same augmentation as the 2-D pipeline (dataset_synapse.RandomGenerator)."""
import os
import random
import re
from collections import defaultdict

import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset


class StackRandomGenerator(object):
    """RandomGenerator for (S, H, W) image stacks and an (H, W) centre label: one set of random draws per sample."""

    def __init__(self, output_size, aug="default"):
        self.output_size = output_size
        self.aug = aug

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        def rotate(image, label):
            angle = np.random.randint(-20, 20)
            return (ndimage.rotate(image, angle, axes=(2, 1), order=0, reshape=False),
                    ndimage.rotate(label, angle, order=0, reshape=False))

        if self.aug == "small_rot":
            if random.random() > 0.5:
                image, label = rotate(image, label)
        elif random.random() > 0.5:
            k = np.random.randint(0, 4)
            image, label = np.rot90(image, k, axes=(1, 2)), np.rot90(label, k)
            if self.aug != "rot_only":
                axis = np.random.randint(0, 2)
                image, label = np.flip(image, axis=axis + 1), np.flip(label, axis=axis)
            image, label = image.copy(), label.copy()
        elif random.random() > 0.5:
            image, label = rotate(image, label)
        _, x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (1, self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        return {"image": torch.from_numpy(image.astype(np.float32)), "label": torch.from_numpy(label.astype(np.float32)).long()}


class SliceStack_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, K, transform=None):
        assert split == "train", "test volumes are read with Synapse_dataset; test_single_volume builds the stacks"
        self.data_dir, self.K, self.transform = base_dir, K, transform
        self.sample_list = [l.strip() for l in open(os.path.join(list_dir, split + ".txt")) if l.strip()]
        self.depth = defaultdict(int)
        for name in self.sample_list:
            case, z = re.match(r"(.*)_slice(\d+)$", name).groups()
            self.depth[case] = max(self.depth[case], int(z) + 1)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        name = self.sample_list[idx]
        case, z = re.match(r"(.*)_slice(\d+)$", name).groups()
        z, width = int(z), len(re.match(r".*_slice(\d+)$", name).group(1))
        images = []
        for dz in range(-self.K, self.K + 1):
            zz = min(max(z + dz, 0), self.depth[case] - 1)
            data = np.load(os.path.join(self.data_dir, f"{case}_slice{zz:0{width}d}.npz"))
            images.append(data["image"])
            if dz == 0:
                label = data["label"]
        sample = {"image": np.stack(images), "label": label}
        if self.transform:
            sample = self.transform(sample)
        sample["case_name"] = name
        return sample
