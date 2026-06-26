"""
Generate train.txt and test_vol.txt for the Kaggle ACDC preprocessed dataset.

Run from inside the ACDC_preprocessed/ directory:
    python /path/to/generate_acdc_lists.py --list_dir /path/to/ApproxDA-TransUNet/lists/lists_ACDC

Split: patient001-070 → train slices (from ACDC_training_slices/)
       patient071-100 → test  volumes (from ACDC_training_volumes/)
"""

import argparse
import glob
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--list_dir', required=True, help='Where to write train.txt and test_vol.txt')
parser.add_argument('--train_end', type=int, default=70, help='Last patient number for training (default: 70)')
args = parser.parse_args()

os.makedirs(args.list_dir, exist_ok=True)

train_entries = []
for path in sorted(glob.glob('ACDC_training_slices/*.h5')):
    name = Path(path).stem
    if '(1)' in name or '(' in name:
        continue
    try:
        patient_num = int(name[len('patient'):len('patient') + 3])
    except ValueError:
        continue
    if patient_num <= args.train_end:
        train_entries.append(name)

test_entries = []
for path in sorted(glob.glob('ACDC_training_volumes/*.h5')):
    name = Path(path).stem
    try:
        patient_num = int(name[len('patient'):len('patient') + 3])
    except ValueError:
        continue
    if patient_num > args.train_end:
        test_entries.append(name)

train_path = os.path.join(args.list_dir, 'train.txt')
test_path  = os.path.join(args.list_dir, 'test_vol.txt')

with open(train_path, 'w') as f:
    f.write('\n'.join(train_entries) + '\n')

with open(test_path, 'w') as f:
    f.write('\n'.join(test_entries) + '\n')

print(f"Train slices : {len(train_entries)}  → {train_path}")
print(f"Test  volumes: {len(test_entries)}  → {test_path}")
print("Sample train:", train_entries[:3])
print("Sample test: ", test_entries[:3])
