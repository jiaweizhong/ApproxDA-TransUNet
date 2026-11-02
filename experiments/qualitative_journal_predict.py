"""
Predict every test case with one trained model, for the journal's qualitative figure (make_qualitative_figure_journal.py).

Run from inside a project directory (ApproxDA-TransUNet or DA-TransUNet) with that project's test.py arguments. The model
and the test set are built by the project's own test.py (its __main__ block up to the logging setup, so no test log is
written), and the checkpoint is the one test.py would evaluate. Prediction follows utils.test_single_volume exactly:
each slice / image is resized to img_size (order 3), argmax, then resized back to the label size (order 0).

Writes <out_dir>/<case>.npz with `pred` (uint8) and <out_dir>/dice.json with the per-case Dice
(mean over the foreground classes present in the ground truth).

usage (from experiments/<project>): python ../qualitative_journal_predict.py OUT_DIR -- <test.py args>
"""

import json
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.ndimage import zoom  # noqa: E402

out_dir = sys.argv[1]
argv = sys.argv[2:]
argv = argv[1:] if argv[:1] == ["--"] else argv
src = open("test.py").read()
cut = src.index('    log_folder = "./test_log/test_log_"')
sys.argv = ["test.py"] + argv
ns = {"__name__": "__main__"}
exec(compile(src[:cut], "test.py", "exec"), ns)
args, net = ns["args"], ns["net"].eval()
db = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir,
                  **({"rgb": True} if getattr(args, "rgb", False) else {}))
S = args.img_size


@torch.no_grad()
def predict_2d(img):
    """img (H, W) or (H, W, 3) -> (H, W) labels, as utils.test_single_volume."""
    x, y = img.shape[:2]
    if (x, y) != (S, S):
        img = zoom(img, (S / x, S / y) + (1,) * (img.ndim - 2), order=3)
    t = torch.from_numpy(img).float()
    t = (t.permute(2, 0, 1) if t.ndim == 3 else t.unsqueeze(0)).unsqueeze(0).cuda()
    out = torch.argmax(torch.softmax(net(t), dim=1), dim=1).squeeze(0).cpu().numpy()
    return zoom(out, (x / S, y / S), order=0) if (x, y) != (S, S) else out


def dice(pred, label):
    ds = []
    for c in np.unique(label):
        if c == 0:
            continue
        p, g = pred == c, label == c
        ds.append(2 * (p & g).sum() / (p.sum() + g.sum()))
    return float(np.mean(ds)) if ds else float("nan")


os.makedirs(out_dir, exist_ok=True)
scores = {}
for i in range(len(db)):
    s = db[i]
    image, label, case = np.asarray(s["image"]), np.asarray(s["label"]), s["case_name"]
    if label.ndim == 3:  # volume (D, H, W): slice by slice
        pred = np.stack([predict_2d(image[k]) for k in range(image.shape[0])])
    else:
        pred = predict_2d(image)
    np.savez_compressed(os.path.join(out_dir, case + ".npz"), pred=pred.astype(np.uint8))
    scores[case] = dice(pred, label)
    print(f"{case} {scores[case] * 100:.2f}", flush=True)
json.dump(scores, open(os.path.join(out_dir, "dice.json"), "w"), indent=1)
print(f"mean {np.nanmean(list(scores.values())) * 100:.2f} over {len(scores)} cases")
