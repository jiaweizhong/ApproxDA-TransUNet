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
from utils import test_single_volume
from Architecture.DATransUNet import DA_Transformer as ViT_seg
from Architecture.DATransUNet import CONFIGS as CONFIGS_ViT_seg

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
    "--ablate_cam",
    type=str,
    default="",
    choices=["", "none", "all", "bottleneck", "skip"],
    help="set CAM_Module gamma to 0 (CAM returns its input) at the bottleneck, the skips or both (log suffix _cabl_<where>)",
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
    "(needs lists/<dataset>/val.txt, e.g. ISIC official val)",
)
args = parser.parse_args()


def precise_bn(args, net):
    """Precise BN: re-estimate every BatchNorm layer's running statistics as a cumulative average over one
    seeded pass of the augmented training set (weights untouched). DANetHead-style BN uses momentum 0.95, so the
    saved statistics are essentially those of the last training mini-batch of the final epoch (a single image
    on the ISIC 75/25 split), which adds evaluation noise of up to ~1.6 Dice points."""
    import importlib
    from torchvision import transforms

    train_root = {"Synapse": "../data/Synapse/train_npz", "ACDC": "../data/ACDC/ACDC_training_slices"}
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
    """AMOS: mean of each class over the cases that contain it (nnU-Net convention), then the class average, for all
    15 organs and for the eight Synapse organs; HD95 in mm. Hallucinated organs (absent from the ground truth but
    predicted) and missed organs (present but not predicted) are counted separately."""
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
            metric_mode=metric_mode,
            spacing=sampled_batch["spacing"][0].numpy() if "spacing" in sampled_batch else None,
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
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]["num_classes"]
    args.volume_path = dataset_config[dataset_name]["volume_path"]
    args.Dataset = dataset_config[dataset_name]["Dataset"]
    args.list_dir = dataset_config[dataset_name]["list_dir"]
    args.z_spacing = dataset_config[dataset_name]["z_spacing"]
    args.is_pretrain = True

    # name the same snapshot defined in train script!
    args.exp = "TU_" + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, "TU")
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
    snapshot_path = snapshot_path + "_adam" if args.optimizer == "adam" else snapshot_path
    snapshot_path = snapshot_path + "_rgb" if args.rgb else snapshot_path
    snapshot_path = snapshot_path + f"_nw{args.num_workers}" if args.num_workers != 4 else snapshot_path
    snapshot_path = snapshot_path + "_dl" if args.drop_last else snapshot_path

    if args.split == "val" and not os.path.isfile(os.path.join(args.list_dir, "val.txt")):
        raise SystemExit(f"--split val: {args.list_dir}/val.txt not found ({dataset_name} has no val split)")

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find("R50") != -1:
        config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size),
            int(args.img_size / args.vit_patches_size),
        )
    net = ViT_seg(
        config_vit, img_size=args.img_size, num_classes=config_vit.n_classes
    ).cuda()

    # Protocol: evaluate the final-epoch checkpoint. best_model.pth is only used on
    # request (it may be a stale file from older runs that validated on the test set).
    last = args.eval_epoch if args.eval_epoch > 0 else args.max_epochs
    ckpt_name = "best_model.pth" if args.checkpoint == "best" else f"epoch_{last - 1}.pth"
    snapshot = os.path.join(snapshot_path, ckpt_name)
    if not os.path.exists(snapshot):
        raise SystemExit(f"checkpoint not found: {snapshot}")
    state_dict = torch.load(snapshot)
    # epoch_*.pth is saved from the (possibly DataParallel-wrapped) model -> strip "module."
    state_dict = {
        k[len("module."):] if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }
    net.load_state_dict(state_dict)
    snapshot_name = snapshot_path.split("/")[-1]
    # keep val-split and best-checkpoint logs separate from the final test logs
    snapshot_name += "_val" if args.split == "val" else ""
    snapshot_name += "_best" if args.checkpoint == "best" else ""
    snapshot_name += f"_ep{args.eval_epoch}" if args.eval_epoch > 0 else ""
    snapshot_name += "_pbn" if args.precise_bn else ""
    if args.ablate_cam:
        n_off = 0
        for name, m in net.named_modules():
            if type(m).__name__ == "CAM_Module":
                where = "bottleneck" if name.startswith("transformer.embeddings.") else "skip"
                if args.ablate_cam in ("all", where):  # "none": same run, separate log (sanity check)
                    m.gamma.data.zero_()
                    n_off += 1
        print(f"--ablate_cam {args.ablate_cam}: {n_off} CAM modules switched off")
        snapshot_name += f"_cabl_{args.ablate_cam}"

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
    if args.precise_bn:
        precise_bn(args, net)
    inference(args, net, test_save_path)
