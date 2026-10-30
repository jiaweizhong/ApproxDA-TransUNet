import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from datasets.dataset_acdc import ACDC_dataset
from datasets.dataset_amos import AMOS_dataset
from datasets.dataset_feta import FeTA_dataset
from datasets.dataset_kvasir import Kvasir_dataset
from datasets.dataset_isic import ISIC_dataset
from datasets.dataset_cvc import CVC_dataset
from datasets.dataset_kvasir_instrument import KvasirInstrument_dataset
from datasets.dataset_chest_xray import ChestXray_dataset
from utils import test_single_volume
from Architecture.ApproxDATransUNet import ApproxDATransUNet
from Architecture.ApproxDATransUNet import CONFIGS as CONFIGS_ViT_seg

try:
    from fvcore.nn import FlopCountAnalysis

    _FLOP_LIB = "fvcore"
except ImportError:
    try:
        from thop import profile as thop_profile

        _FLOP_LIB = "thop"
    except ImportError:
        _FLOP_LIB = None

parser = argparse.ArgumentParser()
parser.add_argument(
    "--volume_path",
    type=str,
    default="../data/Synapse/test_vol_h5",
    help="root dir for validation volume data",
)  # for acdc volume_path=root_dir
parser.add_argument("--dataset", type=str, default="Synapse", help="experiment_name")
parser.add_argument(
    "--num_classes", type=int, default=4, help="output channel of network"
)
parser.add_argument(
    "--list_dir", type=str, default="./lists/lists_Synapse", help="list dir"
)

parser.add_argument(
    "--max_iterations", type=int, default=20000, help="maximum epoch number to train"
)
parser.add_argument(
    "--max_epochs", type=int, default=150, help="maximum epoch number to train"
)
parser.add_argument("--batch_size", type=int, default=24, help="batch_size per gpu")
parser.add_argument(
    "--img_size", type=int, default=224, help="input patch size of network input"
)
parser.add_argument(
    "--is_savenii", action="store_true", help="whether to save results during inference"
)

parser.add_argument(
    "--n_skip", type=int, default=3, help="using number of skip-connect, default is num"
)
parser.add_argument(
    "--vit_name", type=str, default="R50-ViT-B_16", help="select one vit model (must match training)"
)

parser.add_argument(
    "--test_save_dir",
    type=str,
    default="../predictions",
    help="saving prediction as nii!",
)
parser.add_argument(
    "--deterministic", type=int, default=1, help="whether use deterministic training"
)
parser.add_argument(
    "--base_lr", type=float, default=0.01, help="segmentation network learning rate"
)
parser.add_argument("--seed", type=int, default=1234, help="random seed")
parser.add_argument(
    "--vit_patches_size", type=int, default=16, help="vit_patches_size, default is 16"
)
parser.add_argument(
    "--window_size", type=int, default=7, help="window size — must match training"
)
parser.add_argument(
    "--rank", type=int, default=32, help="low-rank dim — must match training"
)
parser.add_argument(
    "--groups", type=int, default=8, help="channel groups — must match training"
)
parser.add_argument(
    "--gate_mode",
    type=str,
    default="learn",
    choices=["learn", "fixed", "pam", "cam"],
    help="Must match training gate_mode",
)
parser.add_argument(
    "--block_version",
    type=str,
    default="c16",
    choices=["c16", "legacy"],
    help="Must match training block_version (c16 = DANetHead-style C/16 bottleneck)",
)
parser.add_argument("--alpha_init", type=float, default=0.0,
                    help="initial PAM scale alpha (default 0 as DANet; snapshot suffix _a<value>)")
parser.add_argument("--ablate_attn", action="store_true",
                    help="switch every PAM off at test time (returns its input) to measure how much the model uses "
                    "attention (log suffix _abl)")
parser.add_argument("--ablate_attn_at", type=str, default="all", choices=["all", "bottleneck", "skip"],
                    help="with --ablate_attn: switch off the PAM of every block, only the bottleneck block or only the skip "
                    "blocks (log suffix _abl / _abl_bottleneck / _abl_skip)")
parser.add_argument("--lr_post", action="store_true",
                    help="Synapse: merge the predicted kidneys and assign left/right per 3-D component by its side of the "
                    "predicted aorta (utils.relabel_kidneys_by_side; log suffix _lrpost)")
parser.add_argument("--upsample", type=str, default="", choices=["", "nearest", "bilinear", "refine"],
                    help="how 224 predictions reach the native slice size: nearest = argmax then nearest zoom (TransUNet, "
                    "default), bilinear = bilinear logits then argmax, refine = bilinear + point head (default with "
                    "--point_refine); a non-default choice gets the log suffix _up<mode>")
parser.add_argument("--ablate_third", action="store_true",
                    help="--third_branch models: zero the organ context / triplet output of the third branch at test time "
                    "(its f path stays; log suffix _tabl)")
parser.add_argument("--ablate_slice", type=str, default="", choices=["", "all", "neighbours"],
                    help="--slice_attn models: skip the slice attention (all) or restrict its keys to the centre slice "
                    "(neighbours) at test time (log suffix _sabl_<mode>)")
parser.add_argument("--ablate_cam", type=str, default="", choices=["", "none", "all", "bottleneck", "skip"],
                    help="set CAM beta to 0 (CAM returns its input) at the bottleneck, the skips or both, to measure "
                    "how much the model uses channel attention (log suffix _cabl_<where>)")
parser.add_argument("--backbone", type=str, default="vit", choices=["vit", "lightvit_t", "lightvit_s", "lightvit_b", "conv"],
                    help="vit = R50 + ViT-B/16 (default); lightvit_* = R50 up to 1/8 + LightViT stages (suffix _lvt/_lvs/_lvb); "
                    "conv = R50 up to 1/8 + one 3x3 stride-2 conv to the LightViT-T bottleneck shape, no transformer (suffix _cnv)")
parser.add_argument("--lightvit_dir", type=str, default="/data/checkpoints/LightViT",
                    help="directory with lightvit_{tiny,small,base}.pth (ImageNet-1k)")
parser.add_argument("--no_vit", action="store_true",
                    help="drop the 12 ViT-B layers (R50 1/16 features -> patch embedding -> decoder); measures the "
                    "global-context contribution of the transformer (snapshot suffix _novit)")
parser.add_argument("--dec_channels", type=int, default=0,
                    help="EffiDec-style slim decoder with this many channels per stage (0 = original; suffix _dec<C>)")
parser.add_argument("--dec_stop112", action="store_true",
                    help="slim decoder only: drop the full-resolution stage, predict at 1/2 and upsample (suffix _s112)")
parser.add_argument("--pam_style", type=str, default="danet", choices=["danet", "tf"],
                    help="danet = DANet PAM (alpha*E + x, default); tf = transformer-style PAM (snapshot suffix _tf)")
parser.add_argument("--pam_fusion", type=str, default="residual", choices=["residual", "gate", "max", "relu_add", "lse"],
                    help="tf PAM: LayerScale residual (default) or per-pixel gate between x and E (suffix _fg); DANet PAM: "
                    "OR-like fusions with the identity as a floor, max(x,E) / x+relu(E) / log(e^x+e^E) (suffix _fmax/_fra/_flse)")
parser.add_argument("--attn_off", action="store_true",
                    help="conv-only head: PAM passes its input through (snapshot suffix _aoff)")
parser.add_argument("--attn_opt", type=str, default="same", choices=["same", "adamw"],
                    help="adamw: optimize the PAM/CAM parameters with AdamW, the rest with --optimizer "
                    "(snapshot suffix _aw<attn_lr>)")
parser.add_argument("--attn_lr", type=float, default=1e-3, help="--attn_opt adamw: base lr of the attention parameters")
parser.add_argument("--attn_scope", type=str, default="pamcam", choices=["pamcam", "added"],
                    help="--attn_opt adamw: which parameters get AdamW: the PAM/CAM of every block (default) or only the "
                    "added modules, --slice_attn and --third_branch (snapshot suffix _awadd<attn_lr>)")
parser.add_argument("--attn_wd", type=float, default=0.01,
                    help="--attn_opt adamw: weight decay of the attention weight matrices (scales, norms and biases: 0)")
parser.add_argument("--attn_off_at", type=str, default="all", choices=["all", "bottleneck", "skip"],
                    help="with --attn_off: conv-only head everywhere, only at the bottleneck block or only at the skip blocks "
                    "(snapshot suffix _aoff / _aoffbn / _aoffsk)")
parser.add_argument("--no_block", action="store_true",
                    help="no ApproxDABlocks at all, TransUNet-style skips (snapshot suffix _noblk)")
parser.add_argument("--aug", type=str, default="default", choices=["default", "rot_only", "small_rot"],
                    help="Synapse training augmentation: default = TransUNet rot90+flip (mirrors half the slices without "
                    "swapping the kidney labels); rot_only = rot90 without flip; small_rot = only +-20 deg rotations "
                    "(snapshot suffix _augrot / _augsmall)")
parser.add_argument("--pam_relpos", action="store_true",
                    help="DANet PAM: Swin-style relative position bias in the logits (snapshot suffix _rpb)")
parser.add_argument("--pam_abspos", action="store_true",
                    help="DANet PAM: normalised global (y, x) coordinates appended to the query/key input (suffix _apos)")
parser.add_argument("--coord_gate", action="store_true",
                    help="gate=pam: coordinate-attention residual gate on the PAM output, identity at init (suffix _cg)")
parser.add_argument("--bn_order", type=str, default="post", choices=["post", "pre"],
                    help="--backbone lightvit_*: bottleneck ApproxDA block after LightViT (post, default) or on R50's 1/8 map "
                    "before LightViT (pre, the ViT-B order) (snapshot suffix _bnpre)")
parser.add_argument("--hr_skip", type=int, default=0,
                    help="slim decoder: H-channel full-resolution feature from the input image, concatenated in the "
                    "full-resolution stage (U-Net-style top skip; suffix _hr<H>)")
parser.add_argument("--final_channels", type=int, default=16,
                    help="slim decoder: channels of the full-resolution stage (default 16; suffix _fc<F>)")
parser.add_argument("--point_loss", action="store_true",
                    help="training-only native-resolution point supervision on the coarse logits (PointRend point "
                    "selection, no point head, nothing at inference) (snapshot suffix _ptl)")
parser.add_argument("--ptl_balanced", action="store_true",
                    help="with --point_loss: 25%% of the points spread evenly over the classes present in the image, "
                    "so small organs always get points (snapshot suffix _ptlb)")
parser.add_argument("--ptl_class_share", type=float, default=0.25,
                    help="with --ptl_balanced: share of the points spread over the classes present "
                    "(snapshot suffix _ptlb<share> when it is not 0.25)")
parser.add_argument("--point_refine", action="store_true",
                    help="PointRend-style native-resolution boundary refinement head (point MLP on coarse logits, decoder "
                    "features and a 5x5 native intensity patch at the most uncertain points); Synapse / AMOS only (suffix _pr)")
parser.add_argument("--pr_residual", action="store_true",
                    help="--point_refine: residual point head (coarse logits + MLP, last layer zero-initialised) (suffix r)")
parser.add_argument("--pr_balanced", action="store_true",
                    help="--point_refine: 25%% of the training points spread over the classes present and a 0.5 CE + 0.5 "
                    "Dice point loss (suffix b)")
parser.add_argument("--slice_attn", type=int, default=-1,
                    help="-1 off; K >= 0: inter-slice attention over the bottleneck tokens of the 2K+1 adjacent slices "
                    "(shared encoder, centre slice decoded; K = 0 is the same layer on the centre slice only, the control); "
                    "Synapse / AMOS only (snapshot suffix _sa<K>)")
parser.add_argument("--third_branch", type=str, default="none", choices=["none", "organ", "triplet"],
                    help="third branch of the bottleneck block, added at full width: organ = organ-level attention with "
                    "an auxiliary coarse segmentation (OCR-style + organ-organ relations); triplet = unsupervised triplet "
                    "attention control; Synapse / AMOS only (snapshot suffix _3organ / _3triplet)")
parser.add_argument("--organ_aux", type=float, default=0.4,
                    help="--third_branch organ: weight of the auxiliary coarse-segmentation loss (suffix _oa<w> if not 0.4)")
parser.add_argument("--pam_affinity", type=float, default=0.0,
                    help="weight of the Context Prior affinity loss on the bottleneck PAM (queries/keys from the block input, "
                    "sigmoid affinity, exact global attention); Synapse / AMOS only (snapshot suffix _aff<w>)")
parser.add_argument("--refine_mode", type=str, default="plain", choices=["plain", "dilated", "guided"],
                    help="--block_type refine: plain DANetHead conv path / dilated (1,2,4) middle conv / decoder-guided "
                    "skip refinement (snapshot suffix _rf<mode>)")
parser.add_argument("--block_type", type=str, default="approxda", choices=["approxda", "mednext", "refine"],
                    help="mednext: replace every ApproxDABlock (bottleneck and skips) with a MedNeXt-style residual conv "
                    "block (snapshot suffix _mnx, plus k<k>e<exp> when not 5 / 2)")
parser.add_argument("--mednext_k", type=int, default=5, help="--block_type mednext: depthwise kernel size")
parser.add_argument("--mednext_exp", type=int, default=2, help="--block_type mednext: expansion ratio of the 1x1 MLP")
parser.add_argument("--skip_ratio", type=int, default=16,
                    help="skip-connection ApproxDABlocks compress C -> C/skip_ratio (DA: 16; bottleneck block stays C/16; "
                    "snapshot suffix _sr<r>)")
parser.add_argument("--pam_no_identity", action="store_true",
                    help="PAM outputs only alpha*E (no +x path), so attention is the only route (snapshot suffix _ni)")
parser.add_argument("--attn_norm", action="store_true",
                    help="BatchNorm on the PAM attention term before scaling by alpha (snapshot suffix _an)")
parser.add_argument(
    "--run_tag",
    type=str,
    default="",
    help="suffix for a repeat of an existing run (snapshot suffix _<tag>), so the original is not overwritten",
)
parser.add_argument(
    "--drop_last",
    action="store_true",
    help="drop the incomplete last training batch of each epoch (snapshot suffix _dl); with BN momentum "
    "0.95 a 1-image last batch (ISIC 75/25) dominates the running statistics",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    help="training DataLoader workers (4 = all runs so far); a different value changes the augmentation "
    "RNG stream, so it gets a _nw<N> snapshot suffix",
)
parser.add_argument(
    "--rgb",
    action="store_true",
    help="Kvasir/ISIC only: feed RGB images instead of grayscale (snapshot suffix _rgb)",
)
parser.add_argument(
    "--optimizer",
    type=str,
    default="sgd",
    choices=["sgd", "adam"],
    help="Must match training optimizer (selects the snapshot directory)",
)
parser.add_argument(
    "--precise_bn",
    action="store_true",
    help="re-estimate BatchNorm statistics on the training set before evaluating (log suffix _pbn)",
)
parser.add_argument(
    "--eval_epoch",
    type=int,
    default=0,
    help="evaluate the intermediate checkpoint epoch_<N-1>.pth of this run instead of the final epoch "
    "(0 = final epoch); the poly LR schedule has not finished there",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="last",
    choices=["last", "best"],
    help="last = final-epoch checkpoint (protocol default); best = best_model.pth "
    "saved by val-split validation",
)
parser.add_argument(
    "--split",
    type=str,
    default="test",
    choices=["test", "val"],
    help="evaluate on the test split (final evaluation) or the val split "
    "(hyperparameter selection; needs lists/<dataset>/val.txt, e.g. ISIC official val)",
)
args = parser.parse_args()


def precise_bn(args, net):
    """Precise BN: re-estimate every BatchNorm layer's running statistics as a cumulative average over one
    seeded pass of the augmented training set (weights untouched). DANetHead-style BN uses momentum 0.95, so the
    saved statistics are essentially those of the last training mini-batch of the final epoch (a single image
    on the ISIC 75/25 split), which adds evaluation noise of up to ~1.6 Dice points."""
    import importlib
    from torchvision import transforms

    train_root = {"Synapse": "../data/Synapse/train_npz", "ACDC": "../data/ACDC_MTUNet", "AMOS": "../data/AMOS_CT_2d/train_npz",
                  "FeTA": "../data/FeTA_2d/train_npz"}
    gen = importlib.import_module(args.Dataset.__module__).RandomGenerator([args.img_size, args.img_size])
    db = args.Dataset(
        base_dir=train_root.get(args.dataset, args.volume_path),
        list_dir=args.list_dir,
        split="train",
        transform=transforms.Compose([gen]),
        **({"rgb": True} if args.rgb else {}),
    )
    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = DataLoader(db, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True, generator=g)
    bns = [m for m in net.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative average
    net.eval()
    for m in bns:
        m.train()
    with torch.no_grad():
        for batch in loader:
            net(batch["image"].cuda())
    net.eval()
    logging.info("precise BN: %d BatchNorm layers re-estimated on %d training batches" % (len(bns), len(loader)))


def log_absent_aware_summary(args, per_case):
    """AMOS / FeTA: mean of each class over the cases that contain it (nnU-Net convention), then the class average,
    for all classes (AMOS: also for the eight Synapse organs); HD95 in mm. Hallucinated organs (absent from the ground
    truth but predicted) and missed organs (present but not predicted) are counted separately."""
    import warnings
    from datasets.dataset_amos import AMOS_CLASSES, SYNAPSE8
    from datasets.dataset_feta import FETA_CLASSES

    if args.dataset == "FeTA":
        CLASSES, groups = FETA_CLASSES, (("all7", list(range(len(FETA_CLASSES)))),)
    else:
        CLASSES = AMOS_CLASSES
        groups = (("synapse8", [c - 1 for c in SYNAPSE8]), ("all15", list(range(len(AMOS_CLASSES)))))
    M = np.stack([m for _, m in per_case])  # (cases, classes, 5): dice, hd95, iou, predicted voxels, true voxels
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cls = np.nanmean(M[:, :, :3], axis=0)
    for i, name in enumerate(CLASSES):
        absent = M[:, i, 4] == 0
        logging.info(
            "Mean class %d %s mean_dice %f mean_hd95 %f mean_iou %f (cases %d, missed %d, hallucinated %d/%d)"
            % (i + 1, name, *cls[i], (~absent).sum(), (~absent & (M[:, i, 3] == 0)).sum(),
               (absent & (M[:, i, 3] > 0)).sum(), absent.sum())
        )
    hall = [(c, CLASSES[i], int(m[i, 3])) for c, m in per_case for i in range(len(CLASSES)) if m[i, 4] == 0 and m[i, 3] > 0]
    logging.info("hallucinated organs (case, class, predicted voxels): %s" % hall)
    n_missed = int(((M[:, :, 4] > 0) & (M[:, :, 3] == 0)).sum())
    logging.info("missed organs (present, not predicted; Dice 0, excluded from HD95): %d case-class pairs" % n_missed)
    for name, idx in groups:
        dice, iou = cls[idx, 0].mean(), cls[idx, 2].mean()
        with warnings.catch_warnings():  # a class missed in every case has no HD95
            warnings.simplefilter("ignore", RuntimeWarning)
            hd95 = np.nanmean(cls[idx, 1])
        logging.info(
            "Testing performance [%s] (split=%s, checkpoint=%s): mean_dice : %f mean_hd95 : %f mean_iou : %f"
            % (name, args.split, args.checkpoint, dice, hd95, iou)
        )


def inference(args, model, test_save_path=None):
    db_test = args.Dataset(
        base_dir=args.volume_path,
        split="val" if args.split == "val" else "test_vol",
        list_dir=args.list_dir,
        **({"rgb": True} if args.rgb else {}),
    )
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info("Model parameters: {:.2f} M".format(n_params))

    # eval() before profiling: the FLOP-count forward pass would otherwise run in
    # train mode and overwrite BatchNorm running stats (DANetHead uses momentum 0.95)
    model.eval()
    if _FLOP_LIB:
        try:
            dummy = torch.randn(1, 1, args.img_size, args.img_size).cuda()
            if _FLOP_LIB == "fvcore":
                _fa = FlopCountAnalysis(model, dummy)
                _fa.unsupported_ops_warnings(False)
                _fa.uncalled_modules_warnings(False)
                logging.info(
                    "GFLOPs (single 224x224 slice, fvcore, includes attention bmm): {:.1f}".format(
                        _fa.total() / 1e9
                    )
                )
            else:
                macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
                logging.info(
                    "GFLOPs (single 224x224 slice, thop, attention bmm NOT counted): {:.1f}".format(
                        macs / 1e9
                    )
                )
        except Exception as e:
            logging.info("GFLOPs: could not profile ({})".format(e))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    metric_list = 0.0
    # AMOS has organs absent from some cases; FeTA uses the same metric for HD95 in mm (spacing differs per subject)
    metric_mode = "nan" if args.dataset in ("AMOS", "FeTA") else "legacy"
    per_case = []
    infer_start = time.time()
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, case_name = (
            sampled_batch["image"],
            sampled_batch["label"],
            sampled_batch["case_name"][0],
        )
        metric_i = test_single_volume(
            image,
            label,
            model,
            classes=args.num_classes,
            patch_size=[args.img_size, args.img_size],
            test_save_path=test_save_path,
            case=case_name,
            z_spacing=args.z_spacing,
            lr_post=args.lr_post,
            metric_mode=metric_mode,
            spacing=sampled_batch["spacing"][0].numpy() if "spacing" in sampled_batch else None,
            pred_save_path=os.path.join(args.pred_dir, case_name + ".npz") if args.pred_dir else None,
            slice_ctx=args.slice_attn,
            upsample=args.upsample,
        )
        if metric_mode == "nan":
            m = np.array(metric_i, dtype=float)
            per_case.append((case_name, m))
            logging.info("idx %d case %s mean_dice %f mean_hd95 %f"
                         % (i_batch, case_name, np.nanmean(m[:, 0]), np.nanmean(m[:, 1])))
            continue
        metric_list += np.array(metric_i)
        logging.info(
            "idx %d case %s mean_dice %f mean_hd95 %f"
            % (
                i_batch,
                case_name,
                np.mean(metric_i, axis=0)[0],
                np.mean(metric_i, axis=0)[1],
            )
        )

    infer_total_s = time.time() - infer_start
    avg_case_ms = infer_total_s / len(db_test) * 1000
    logging.info(
        "Inference total: {:.1f}s  avg per case: {:.1f}ms".format(
            infer_total_s, avg_case_ms
        )
    )
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        logging.info(
            "Inference peak VRAM: {:.0f} MB ({:.1f} GB)".format(peak_mb, peak_mb / 1024)
        )

    if metric_mode == "nan":
        log_absent_aware_summary(args, per_case)
        return "Testing Finished!"
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info(
            "Mean class %d mean_dice %f mean_hd95 %f mean_iou %f"
            % (i, metric_list[i - 1][0], metric_list[i - 1][1], metric_list[i - 1][2])
        )
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    mean_iou = np.mean(metric_list, axis=0)[2]
    logging.info(
        "Testing performance (split=%s, checkpoint=%s): mean_dice : %f mean_hd95 : %f mean_iou : %f"
        % (args.split, args.checkpoint, performance, mean_hd95, mean_iou)
    )
    return "Testing Finished!"


if __name__ == "__main__":

    if args.is_savenii:
        print("Saving results as NIfTI files.")
    else:
        print("Not saving results.")

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        "Synapse": {
            "Dataset": Synapse_dataset,
            "volume_path": "../data/Synapse/test_vol_h5",
            "list_dir": "./lists/lists_Synapse",
            "num_classes": 9,
            "z_spacing": 1,
        },
        "AMOS": {
            "Dataset": AMOS_dataset,
            "volume_path": "../data/AMOS_CT_2d/test_vol_h5",
            "list_dir": "./lists/lists_AMOS",
            "num_classes": 16,
            "z_spacing": 1,
        },
        "FeTA": {
            "Dataset": FeTA_dataset,
            "volume_path": "../data/FeTA_2d/test_vol_h5",
            "list_dir": "./lists/lists_FeTA",
            "num_classes": 8,
            "z_spacing": 1,
        },
        "ACDC": {
            "Dataset": ACDC_dataset,
            "volume_path": "../data/ACDC_MTUNet",
            "list_dir": "./lists/lists_ACDC",
            "num_classes": 4,
            "z_spacing": 1,
        },
        "Kvasir": {
            "Dataset": Kvasir_dataset,
            "volume_path": "../data/Kvasir-SEG",
            "list_dir": "./lists/lists_Kvasir",
            "num_classes": 2,
            "z_spacing": 1,
        },
        "ISIC": {
            "Dataset": ISIC_dataset,
            "volume_path": "../data/ISIC2018",
            "list_dir": "./lists/lists_ISIC",
            "num_classes": 2,
            "z_spacing": 1,
        },
        "CVC": {
            "Dataset": CVC_dataset,
            "volume_path": "../data/CVC-ClinicDB",
            "list_dir": "./lists/lists_CVC",
            "num_classes": 2,
            "z_spacing": 1,
        },
        "KvasirInstrument": {
            "Dataset": KvasirInstrument_dataset,
            "volume_path": "../data/Kvasir-Instrument",
            "list_dir": "./lists/lists_KvasirInstrument",
            "num_classes": 2,
            "z_spacing": 1,
        },
        "ChestXray": {
            "Dataset": ChestXray_dataset,
            "volume_path": "../data/Montgomery",
            "list_dir": "./lists/lists_ChestXray",
            "num_classes": 2,
            "z_spacing": 1,
        },
    }
    dataset_name = args.dataset
    if args.lr_post and dataset_name != "Synapse":
        raise SystemExit("--lr_post uses the Synapse label indices")
    args.num_classes = dataset_config[dataset_name]["num_classes"]
    args.volume_path = dataset_config[dataset_name]["volume_path"]
    args.Dataset = dataset_config[dataset_name]["Dataset"]
    args.list_dir = dataset_config[dataset_name]["list_dir"]
    args.z_spacing = dataset_config[dataset_name]["z_spacing"]
    args.is_pretrain = True

    # name the same snapshot defined in train script!
    args.exp = "ApproxDA_" + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, "ApproxDA")
    snapshot_path = snapshot_path + "_pretrain" if args.is_pretrain else snapshot_path
    snapshot_path += "_" + args.vit_name
    snapshot_path = snapshot_path + "_skip" + str(args.n_skip)
    snapshot_path = (
        snapshot_path + "_vitpatch" + str(args.vit_patches_size)
        if args.vit_patches_size != 16
        else snapshot_path
    )
    snapshot_path = (
        snapshot_path + "_epo" + str(args.max_epochs)
        if args.max_epochs != 30
        else snapshot_path
    )
    # (removed 2026-10-20: an ACDC-only '_20k' suffix that train.py never writes, so test.py missed the checkpoint)
    snapshot_path = snapshot_path + "_bs" + str(args.batch_size)
    snapshot_path = (
        snapshot_path + "_lr" + str(args.base_lr)
        if args.base_lr != 0.01
        else snapshot_path
    )
    snapshot_path = snapshot_path + "_" + str(args.img_size)
    snapshot_path = (
        snapshot_path + "_s" + str(args.seed) if args.seed != 1234 else snapshot_path
    )
    snapshot_path = (
        snapshot_path + "_M" + str(args.window_size)
        if args.window_size != 7
        else snapshot_path
    )
    snapshot_path = (
        snapshot_path + "_r" + str(args.rank) if args.rank != 32 else snapshot_path
    )
    snapshot_path = (
        snapshot_path + "_G" + str(args.groups) if args.groups != 8 else snapshot_path
    )
    snapshot_path = (
        snapshot_path + "_" + args.gate_mode
        if args.gate_mode != "learn"
        else snapshot_path
    )
    # c16 runs get their own directory so they never pick up legacy checkpoints
    snapshot_path = snapshot_path + "_c16" if args.block_version == "c16" else snapshot_path
    snapshot_path = snapshot_path + f"_a{args.alpha_init:g}" if args.alpha_init != 0 else snapshot_path
    snapshot_path = snapshot_path + "_an" if args.attn_norm else snapshot_path
    snapshot_path = snapshot_path + "_ni" if args.pam_no_identity else snapshot_path
    snapshot_path = snapshot_path + "_tf" if args.pam_style == "tf" else snapshot_path
    snapshot_path = snapshot_path + "_fg" if args.pam_fusion == "gate" else snapshot_path
    snapshot_path = snapshot_path + {"max": "_fmax", "relu_add": "_fra", "lse": "_flse"}.get(args.pam_fusion, "")
    snapshot_path += ("_rpb" if args.pam_relpos else "") + ("_apos" if args.pam_abspos else "") + ("_cg" if args.coord_gate else "")
    snapshot_path += f"_aff{args.pam_affinity:g}" if args.pam_affinity > 0 else ""
    snapshot_path += "" if args.third_branch == "none" else "_3" + args.third_branch
    snapshot_path += f"_oa{args.organ_aux:g}" if args.third_branch == "organ" and args.organ_aux != 0.4 else ""
    snapshot_path += f"_sa{args.slice_attn}" if args.slice_attn >= 0 else ""
    snapshot_path += "_bnpre" if args.bn_order == "pre" else ""
    snapshot_path += f"_hr{args.hr_skip}" if args.hr_skip > 0 else ""
    snapshot_path += f"_fc{args.final_channels}" if args.final_channels != 16 else ""
    snapshot_path += "_ptl" if args.point_loss else ""
    snapshot_path += "b" if args.ptl_balanced else ""  # _ptlb
    snapshot_path += f"{args.ptl_class_share:g}" if args.ptl_balanced and args.ptl_class_share != 0.25 else ""
    if args.ptl_balanced and not args.point_loss:
        raise SystemExit("--ptl_balanced modifies --point_loss")
    if args.point_loss and args.point_refine:
        raise SystemExit("--point_loss is the head-free version of --point_refine; use one of them")
    if args.bn_order == "pre" and args.backbone == "vit":
        raise SystemExit("--bn_order applies to the LightViT backbones only")
    snapshot_path += ("_pr" + ("r" if args.pr_residual else "") + ("b" if args.pr_balanced else "")) if args.point_refine else ""
    if (args.pr_residual or args.pr_balanced) and not args.point_refine:
        raise SystemExit("--pr_residual / --pr_balanced need --point_refine")
    snapshot_path = snapshot_path + {"all": "_aoff", "bottleneck": "_aoffbn", "skip": "_aoffsk"}[args.attn_off_at] if args.attn_off else snapshot_path
    snapshot_path = snapshot_path + "_noblk" if args.no_block else snapshot_path
    snapshot_path = snapshot_path + f"_sr{args.skip_ratio}" if args.skip_ratio != 16 else snapshot_path
    if args.block_type == "refine":
        snapshot_path += "_rf" + args.refine_mode
    if args.block_type == "mednext":
        snapshot_path += "_mnx" + ("" if (args.mednext_k, args.mednext_exp) == (5, 2) else f"k{args.mednext_k}e{args.mednext_exp}")
    snapshot_path = snapshot_path + {"default": "", "rot_only": "_augrot", "small_rot": "_augsmall"}[args.aug]
    snapshot_path = snapshot_path + {"vit": "", "lightvit_t": "_lvt", "lightvit_s": "_lvs", "lightvit_b": "_lvb", "conv": "_cnv"}[args.backbone]
    snapshot_path = snapshot_path + "_novit" if args.no_vit else snapshot_path
    snapshot_path = snapshot_path + f"_dec{args.dec_channels}" if args.dec_channels > 0 else snapshot_path
    snapshot_path = snapshot_path + "_s112" if args.dec_stop112 else snapshot_path
    snapshot_path = snapshot_path + ("_awadd" if args.attn_scope == "added" else "_aw") + f"{args.attn_lr:g}" if args.attn_opt == "adamw" else snapshot_path
    if args.dec_stop112 and args.dec_channels <= 0:
        raise SystemExit("--dec_stop112 requires --dec_channels > 0")
    snapshot_path = snapshot_path + "_adam" if args.optimizer == "adam" else snapshot_path
    snapshot_path = snapshot_path + "_rgb" if args.rgb else snapshot_path
    snapshot_path = snapshot_path + f"_nw{args.num_workers}" if args.num_workers != 4 else snapshot_path
    snapshot_path = snapshot_path + "_dl" if args.drop_last else snapshot_path
    snapshot_path = snapshot_path + f"_{args.run_tag}" if args.run_tag else snapshot_path

    if args.split == "val" and not os.path.isfile(os.path.join(args.list_dir, "val.txt")):
        raise SystemExit(f"--split val: {args.list_dir}/val.txt not found ({dataset_name} has no val split)")

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.window_size = args.window_size
    config_vit.rank = args.rank
    config_vit.groups = args.groups
    config_vit.gate_mode = args.gate_mode
    config_vit.block_version = args.block_version
    config_vit.alpha_init = args.alpha_init
    config_vit.attn_norm = args.attn_norm
    config_vit.pam_no_identity = args.pam_no_identity
    config_vit.pam_style = args.pam_style
    config_vit.pam_fusion = args.pam_fusion
    config_vit.attn_off = args.attn_off
    config_vit.attn_off_at = args.attn_off_at
    config_vit.no_block = args.no_block
    config_vit.skip_ratio = args.skip_ratio
    config_vit.block_type = args.block_type
    config_vit.pam_relpos, config_vit.pam_abspos, config_vit.coord_gate = args.pam_relpos, args.pam_abspos, args.coord_gate
    config_vit.pam_affinity = args.pam_affinity
    config_vit.third_branch = args.third_branch
    config_vit.slice_attn = args.slice_attn
    config_vit.point_refine = args.point_refine
    config_vit.bn_order = args.bn_order
    config_vit.pr_residual, config_vit.pr_balanced = args.pr_residual, args.pr_balanced
    config_vit.refine_mode = args.refine_mode
    config_vit.mednext_k = args.mednext_k
    config_vit.mednext_exp = args.mednext_exp
    config_vit.backbone = args.backbone
    config_vit.lightvit_ckpt = os.path.join(
        args.lightvit_dir, {"vit": "", "lightvit_t": "lightvit_tiny.pth", "lightvit_s": "lightvit_small.pth",
                            "lightvit_b": "lightvit_base.pth", "conv": ""}[args.backbone])
    config_vit.dec_channels = args.dec_channels
    config_vit.hr_skip, config_vit.final_channels = args.hr_skip, args.final_channels
    if args.no_vit:  # Encoder builds no transformer layers; patch embedding, encoder_norm and decoder are unchanged
        config_vit.transformer.num_layers = 0
    config_vit.dec_stop112 = args.dec_stop112
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find("R50") != -1:
        config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size),
            int(args.img_size / args.vit_patches_size),
        )
    net = ApproxDATransUNet(
        config_vit, img_size=args.img_size, num_classes=config_vit.n_classes
    ).cuda()

    # Protocol: evaluate the final-epoch checkpoint. best_model.pth is only used on
    # request (it may be a stale file from older runs that validated on the test set).
    last = args.eval_epoch if args.eval_epoch > 0 else args.max_epochs
    ckpt_name = "best_model.pth" if args.checkpoint == "best" else f"epoch_{last - 1}.pth"
    snapshot = os.path.join(snapshot_path, ckpt_name)
    if not os.path.exists(snapshot):
        raise SystemExit(f"checkpoint not found: {snapshot}")
    state_dict = torch.load(snapshot, weights_only=False)
    # epoch_*.pth is saved from the (possibly DDP-wrapped) model -> strip "module."
    state_dict = {
        k[len("module."):] if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }
    state_dict = {
        k.replace(".ada_block.", ".approx_block."): v for k, v in state_dict.items()
    }
    # Legacy checkpoints still contain the never-called decoder blocks that were
    # removed from the model; drop exactly those keys, fail on anything else.
    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    stale = [k for k in unexpected if k.split(".")[0] == "decoder" and ".da" in k]
    if missing or len(stale) != len(unexpected):
        raise SystemExit(
            f"state_dict mismatch: missing={missing[:5]} "
            f"unexpected={[k for k in unexpected if k not in stale][:5]}"
        )
    if stale:
        print(f"Ignored {len(stale)} keys of never-called legacy decoder blocks")
    snapshot_name = snapshot_path.split("/")[-1]
    # keep val-split and best-checkpoint logs separate from the final test logs
    snapshot_name += "_val" if args.split == "val" else ""
    snapshot_name += "_best" if args.checkpoint == "best" else ""
    snapshot_name += f"_ep{args.eval_epoch}" if args.eval_epoch > 0 else ""
    snapshot_name += "_pbn" if args.precise_bn else ""
    if args.ablate_attn:
        n_off = 0
        for name, m in net.named_modules():
            where = "skip" if name.startswith("decoder.") else "bottleneck"
            if hasattr(m, "ablate") and args.ablate_attn_at in ("all", where):
                m.ablate, n_off = True, n_off + 1
        print(f"--ablate_attn ({args.ablate_attn_at}): {n_off} PAM modules switched off")
        snapshot_name += "_abl" if args.ablate_attn_at == "all" else f"_abl_{args.ablate_attn_at}"
    snapshot_name += "_lrpost" if args.lr_post else ""
    if args.ablate_cam:
        n_off = 0
        for name, m in net.named_modules():
            if isinstance(getattr(m, "beta", None), torch.nn.Parameter):
                where = "bottleneck" if name.startswith("transformer.embeddings.") else "skip"
                if args.ablate_cam in ("all", where):  # "none": same run, separate log (sanity check)
                    m.beta.data.zero_()
                    n_off += 1
        print(f"--ablate_cam {args.ablate_cam}: {n_off} CAM modules switched off")
        snapshot_name += f"_cabl_{args.ablate_cam}"
    default_up = "refine" if args.point_refine else "nearest"
    if not args.upsample:
        args.upsample = default_up
    if args.upsample == "refine" and not args.point_refine:
        raise SystemExit("--upsample refine needs a --point_refine model")
    snapshot_name += f"_up{args.upsample}" if args.upsample != default_up else ""
    if args.ablate_third:
        third = [m for m in net.modules() if type(m).__name__ in ("OrganBranch", "TripletBranch")]
        for m in third:
            m.ablate = True
        print(f"--ablate_third: {len(third)} third branches switched off")
        snapshot_name += "_tabl"
    if args.ablate_slice:
        net.slice_attn.ablate = True if args.ablate_slice == "all" else "neighbours"
        print(f"--ablate_slice {args.ablate_slice}")
        snapshot_name += f"_sabl_{args.ablate_slice}"

    log_folder = "./test_log/test_log_" + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(
        filename=log_folder + "/" + snapshot_name + ".txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = "../predictions"
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    # AMOS: always keep the predicted label maps (uint8, ~1 MB per case) for analyze_misidentification.py
    args.pred_dir = os.path.join("../predictions", args.exp, snapshot_name + "_npz") if dataset_name == "AMOS" else None
    if args.pred_dir:
        os.makedirs(args.pred_dir, exist_ok=True)
    if args.precise_bn:
        precise_bn(args, net)
    inference(args, net, test_save_path)
