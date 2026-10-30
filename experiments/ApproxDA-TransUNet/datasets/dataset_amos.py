"""AMOS 2022 CT in the Synapse format written by prepare_amos.py: training slices are read exactly like Synapse; test
volumes also return their voxel spacing (mm, array axis order) for HD95 in mm."""
import os

import h5py

from datasets.dataset_synapse import RandomGenerator, Synapse_dataset  # noqa: F401  (test.py precise_bn imports RandomGenerator from here)

# AMOS labels: 1 spleen, 2 right kidney, 3 left kidney, 4 gallbladder, 5 esophagus, 6 liver, 7 stomach, 8 aorta,
# 9 inferior vena cava, 10 pancreas, 11 right adrenal gland, 12 left adrenal gland, 13 duodenum, 14 bladder,
# 15 prostate/uterus
AMOS_CLASSES = ("spleen", "kidney_R", "kidney_L", "gallbladder", "esophagus", "liver", "stomach", "aorta", "ivc",
                "pancreas", "adrenal_R", "adrenal_L", "duodenum", "bladder", "prostate_uterus")
SYNAPSE8 = (1, 2, 3, 4, 6, 7, 8, 10)  # the eight Synapse organs, in AMOS indices


class AMOS_dataset(Synapse_dataset):
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        if self.split != "train":
            with h5py.File(os.path.join(self.data_dir, sample["case_name"] + ".npy.h5")) as f:
                sample["spacing"] = f["spacing"][:]
        return sample
