"""FeTA 2.4 fetal brain MRI in the Synapse format written by prepare_feta.py (EffiDec3D 64 / 8 / 8 split): training
slices are read exactly like Synapse; test / validation volumes also return their voxel spacing (mm, array axis order,
0.43 - 0.70 mm isotropic depending on the subject) for HD95 in mm."""
from datasets.dataset_amos import AMOS_dataset
from datasets.dataset_synapse import RandomGenerator  # noqa: F401  (test.py precise_bn imports RandomGenerator from here)

# FeTA labels (dseg.tsv): 1 external CSF, 2 grey matter / cortical plate, 3 white matter / subplate, 4 lateral
# ventricles, 5 cerebellum, 6 deep grey matter (thalamus, putamen), 7 brainstem
FETA_CLASSES = ("eCSF", "GM", "WM", "LV", "CBM", "SGM", "BS")


class FeTA_dataset(AMOS_dataset):
    pass
