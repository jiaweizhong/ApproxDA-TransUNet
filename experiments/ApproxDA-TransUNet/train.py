import argparse
import logging
import os
import random
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from Architecture.ApproxDATransUNet import ApproxDATransUNet
from Architecture.ApproxDATransUNet import CONFIGS as CONFIGS_ViT_seg
from Architecture.block import affinity_loss, organ_aux_loss, point_supervision_loss
import sys
import time
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    type=str,
    default="../data/Synapse/train_npz",
    help="root dir for data",
)
parser.add_argument("--dataset", type=str, default="Synapse", help="experiment_name")
parser.add_argument(
    "--list_dir", type=str, default="./lists/lists_Synapse", help="list dir"
)
parser.add_argument(
    "--num_classes", type=int, default=9, help="output channel of network"
)
parser.add_argument(
    "--max_iterations", type=int, default=30000, help="maximum epoch number to train"
)
parser.add_argument(
    "--max_epochs", type=int, default=150, help="maximum epoch number to train"
)
parser.add_argument("--batch_size", type=int, default=24, help="batch_size per gpu")
parser.add_argument("--n_gpu", type=int, default=1, help="total gpu")
parser.add_argument(
    "--deterministic", type=int, default=1, help="whether use deterministic training"
)
parser.add_argument(
    "--base_lr", type=float, default=0.01, help="segmentation network learning rate"
)
parser.add_argument(
    "--img_size", type=int, default=224, help="input patch size of network input"
)
parser.add_argument("--seed", type=int, default=1234, help="random seed")
parser.add_argument(
    "--n_skip", type=int, default=3, help="using number of skip-connect, default is num"
)
parser.add_argument(
    "--vit_name", type=str, default="R50-ViT-B_16", help="select one vit model"
)
parser.add_argument(
    "--vit_patches_size", type=int, default=16, help="vit_patches_size, default is 16"
)
parser.add_argument(
    "--window_size", type=int, default=7, help="window size for LowRankWindowedPAM"
)
parser.add_argument(
    "--rank",
    type=int,
    default=32,
    help="low-rank projection dimension for LowRankWindowedPAM",
)
parser.add_argument(
    "--groups", type=int, default=8, help="number of channel groups for GroupedCAM"
)
parser.add_argument(
    "--gate_mode",
    type=str,
    default="learn",
    choices=["learn", "fixed", "pam", "cam"],
    help="Gate mode: learn=adaptive (default), fixed=0.5 blend, pam=PAM only, cam=CAM only",
)
parser.add_argument(
    "--block_version",
    type=str,
    default="c16",
    choices=["c16", "legacy"],
    help="c16 = DANetHead-style C/16 bottleneck (default); legacy = BIBM full-width block",
)
parser.add_argument("--alpha_init", type=float, default=0.0,
                    help="initial PAM scale alpha (default 0 as DANet; snapshot suffix _a<value>)")
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
    help="sgd (Synapse, lr 0.01) or adam (Kvasir/ISIC, use --base_lr 0.001), as in DA-TransUNet",
)
parser.add_argument(
    "--val_interval",
    type=int,
    default=0,
    help="validate on the val split every N epochs and save best_model.pth "
    "(0 = disabled; requires lists/<dataset>/val.txt — never uses the test set)",
)
args = parser.parse_args()


class OptimizerPair:
    """Steps two optimizers as one (--attn_opt adamw); param_groups spans both for the lr schedule."""

    def __init__(self, *optimizers):
        self.optimizers = optimizers

    @property
    def param_groups(self):
        return [g for o in self.optimizers for g in o.param_groups]

    def zero_grad(self):
        for o in self.optimizers:
            o.zero_grad()

    def step(self):
        for o in self.optimizers:
            o.step()


def is_attn_param(name):
    return ".pam." in name or ".cam." in name


def is_added_param(name):
    """Parameters of the opt-in added modules (--slice_attn, --third_branch), for --attn_scope added."""
    return name.startswith("slice_attn.") or ".third." in name


def build_optimizer(args, model):
    """SGD for Synapse (TransUNet protocol); Adam for the 2D datasets (DA-TransUNet protocol).

    --attn_opt adamw: the PAM/CAM parameters go to a separate AdamW (lr --attn_lr, following the same poly
    schedule via the group's lr_scale), because under SGD their gradients are too small to move them
    (JOURNAL_EXPERIMENTS.md 3.1). Weight matrices get --attn_wd; scales (alpha, beta, ls_gamma), norms and biases 0.
    """
    if args.attn_opt == "adamw":
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        select = is_added_param if args.attn_scope == "added" else is_attn_param
        attn = [(n, p) for n, p in named if select(n)]
        if not attn:
            raise SystemExit(f"--attn_opt adamw: the model has no {args.attn_scope} parameters")
        scale = args.attn_lr / args.base_lr
        attn_opt = optim.AdamW(
            [
                {"params": [p for n, p in attn if p.dim() >= 2], "weight_decay": args.attn_wd, "lr_scale": scale},
                {"params": [p for n, p in attn if p.dim() < 2], "weight_decay": 0.0, "lr_scale": scale},
            ],
            lr=args.attn_lr,
        )
        rest = [p for n, p in named if not select(n)]
        logging.info(f"--attn_opt adamw ({args.attn_scope}): {len(attn)} tensors "
                     f"({sum(p.numel() for _, p in attn)} params) with AdamW lr {args.attn_lr:g}, rest with {args.optimizer}")
        if args.optimizer == "adam":
            rest_opt = optim.Adam(rest, lr=args.base_lr, betas=(0.9, 0.999), weight_decay=0.0001)
        else:
            rest_opt = optim.SGD(rest, lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
        return OptimizerPair(rest_opt, attn_opt)
    params = model.parameters()
    no_decay = [p for n, p in model.named_parameters() if n.endswith("ls_gamma")]
    if no_decay:  # LayerScale gammas (--pam_style tf) are not weight-decayed; otherwise one group as before
        decay = [p for n, p in model.named_parameters() if not n.endswith("ls_gamma")]
        params = [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]
    if args.optimizer == "adam":
        return optim.Adam(params, lr=args.base_lr, betas=(0.9, 0.999), weight_decay=0.0001)
    return optim.SGD(params, lr=args.base_lr, momentum=0.9, weight_decay=0.0001)


def check_val_split(args):
    """Refuse --val_interval unless a real validation list exists (no test-set selection)."""
    if args.val_interval <= 0:
        return
    val_list = os.path.join(args.list_dir, "val.txt")
    if args.dataset in ("Synapse", "ACDC") or not os.path.isfile(val_list):
        raise SystemExit(
            f"--val_interval {args.val_interval}: {args.dataset} has no validation split "
            f"({val_list} not found). Validation would use the test set; "
            "run with --val_interval 0 and evaluate the final epoch."
        )


class GateMonitor:
    """Per-epoch statistics of the learned routing gate g = sigmoid(gate_fc(GAP(x))).

    A forward hook on every gate_mode='learn' block recomputes g from the block input
    under no_grad (training and RNG are unaffected) and accumulates per-channel sums
    over the epoch's training batches. No-op for the other gate modes.
    """

    def __init__(self, model):
        net = model.module if hasattr(model, "module") else model
        self.blocks = [
            (n, m) for n, m in net.named_modules()
            if getattr(m, "gate_mode", None) == "learn" and hasattr(m, "gate_fc")
        ]
        self.sums = {}
        self.history = {n: [] for n, _ in self.blocks}
        for n, m in self.blocks:
            m.register_forward_hook(self._hook(n))

    def _hook(self, name):
        def hook(mod, inputs, output):  # must return None, or it replaces output
            if not mod.training:
                return
            with torch.no_grad():
                g = torch.sigmoid(mod.gate_fc(mod.pool(inputs[0]).flatten(1)))  # (B, C')
                n, s1, s2 = self.sums.get(name, (0, 0.0, 0.0))
                self.sums[name] = (n + g.shape[0], s1 + g.sum(0), s2 + (g * g).sum(0))

        return hook

    def end_epoch(self, epoch, writer=None):
        for name, (n, s1, s2) in self.sums.items():
            mu = s1 / n
            sd = (s2 / n - mu * mu).clamp_min(0).sqrt()  # per channel, across samples
            mu, sd = mu.cpu().numpy(), sd.cpu().numpy()
            self.history[name].append(mu)
            logging.info(
                "gate %s epoch %d: mean %.4f  channel-std %.4f  sample-std %.4f  min %.4f  max %.4f"
                % (name, epoch, mu.mean(), mu.std(), sd.mean(), mu.min(), mu.max())
            )
            if writer is not None:
                writer.add_scalar(f"gate/{name}/mean", mu.mean(), epoch)
                writer.add_scalar(f"gate/{name}/channel_std", mu.std(), epoch)
                writer.add_scalar(f"gate/{name}/sample_std", sd.mean(), epoch)
        self.sums = {}

    def save(self, path):
        """Per-block (epochs, C') array of per-channel mean gate values."""
        hist = {n: np.stack(v) for n, v in self.history.items() if v}
        if hist:
            np.savez(path, **hist)


def _validate_synapse(args, model):
    from datasets.dataset_synapse import Synapse_dataset
    from utils import test_single_volume

    vol_path = os.path.join(os.path.dirname(args.root_path), "test_vol_h5")
    db_val = Synapse_dataset(
        base_dir=vol_path, split="test_vol", list_dir=args.list_dir
    )
    # num_workers=0: torchrun already runs inside a subprocess; spawning DataLoader
    # worker sub-subprocesses causes nested multiprocessing deadlocks on Linux.
    val_loader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)
    net = model.module if hasattr(model, "module") else model
    # eval/train on the unwrapped net — calling model.eval() on the DDP wrapper
    # triggers an internal all-reduce (find_unused_parameters bookkeeping) that
    # non-validating ranks aren't participating in, causing NCCL ALLREDUCE timeout.
    net.eval()
    metric_list = 0.0
    with torch.no_grad():
        for sample in val_loader:
            metric_i = test_single_volume(
                sample["image"],
                sample["label"],
                net,
                classes=args.num_classes,
                patch_size=[args.img_size, args.img_size],
            )
            metric_list += np.array(metric_i)
    net.train()
    return float(np.mean(metric_list / len(db_val), axis=0)[0])


def _validate_2d(args, model, dataset_class):
    """Run validation-split evaluation for 2D image datasets (e.g. ISIC official val)."""
    from utils import test_single_volume

    db_val = dataset_class(
        base_dir=args.root_path, split="val", list_dir=args.list_dir,
        **({"rgb": True} if args.rgb else {}),
    )
    val_loader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)
    net = model.module if hasattr(model, "module") else model
    net.eval()
    metric_list = 0.0
    with torch.no_grad():
        for sample in val_loader:
            metric_i = test_single_volume(
                sample["image"],
                sample["label"],
                net,
                classes=args.num_classes,
                patch_size=[args.img_size, args.img_size],
            )
            metric_list += np.array(metric_i)
    net.train()
    return float(np.mean(metric_list / len(db_val), axis=0)[0])


def _trainer_2d(args, model, snapshot_path, db_train, dataset_class):
    """Generic training loop for 2D image datasets (Kvasir, ISIC)."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    use_ddp = dist.is_initialized() and world_size > 1
    is_main = not use_ddp or local_rank == 0

    if is_main:
        logging.basicConfig(
            filename=snapshot_path + "/log.txt",
            level=logging.INFO,
            format="[%(asctime)s.%(msecs)03d] %(message)s",
            datefmt="%H:%M:%S",
        )
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size

    if is_main:
        print("The length of train set is: {}".format(len(db_train)))
        print(
            "DDP: {} GPU(s), world_size={}".format(
                world_size if use_ddp else 1, world_size
            )
        )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    if use_ddp:
        sampler = DistributedSampler(db_train, shuffle=True)
        trainloader = DataLoader(
            db_train,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=args.drop_last,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    else:
        trainloader = DataLoader(
            db_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=args.drop_last,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )
    model.train()
    epoch_loss_pt = 0.0  # --point_loss
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = build_optimizer(args, model)
    writer = SummaryWriter(snapshot_path + "/log") if is_main else None
    gates = GateMonitor(model)
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    if is_main:
        logging.info(
            "{} iterations per epoch. {} max iterations ".format(
                len(trainloader), max_iterations
            )
        )
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70, disable=not is_main)
    train_start = time.time()
    val_time_s = 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for epoch_num in iterator:
        epoch_loss_sum = 0.0
        epoch_loss_ce_sum = 0.0
        epoch_batches = 0
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            if args.point_loss:
                loss_pt = point_supervision_loss(outputs, sampled_batch["label_hr"].cuda(non_blocking=True),
                                                 class_share=args.ptl_class_share if args.ptl_balanced else 0.0)
                loss = loss + loss_pt
                epoch_loss_pt += loss_pt.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_ * param_group.get("lr_scale", 1.0)
            iter_num = iter_num + 1
            epoch_loss_sum += loss.item()
            epoch_loss_ce_sum += loss_ce.item()
            epoch_batches += 1
            if writer is not None:
                writer.add_scalar("info/lr", lr_, iter_num)
                writer.add_scalar("info/total_loss", loss, iter_num)
                writer.add_scalar("info/loss_ce", loss_ce, iter_num)
                if iter_num % 20 == 0:
                    image = image_batch[0, 0:1, :, :]
                    image = (image - image.min()) / (image.max() - image.min())
                    writer.add_image("train/Image", image, iter_num)
                    outputs_vis = torch.argmax(
                        torch.softmax(outputs, dim=1), dim=1, keepdim=True
                    )
                    writer.add_image(
                        "train/Prediction", outputs_vis[0, ...] * 127, iter_num
                    )
                    labs = label_batch[0, ...].unsqueeze(0) * 127
                    writer.add_image("train/GroundTruth", labs, iter_num)

        if is_main:
            elapsed_h = (time.time() - train_start - val_time_s) / 3600
            avg_loss = epoch_loss_sum / epoch_batches
            avg_loss_ce = epoch_loss_ce_sum / epoch_batches
            if args.point_loss:
                logging.info("epoch %d/%d  loss_point: %.4f" % (epoch_num + 1, max_epoch, epoch_loss_pt / epoch_batches))
                epoch_loss_pt = 0.0
            logging.info(
                "epoch %d/%d  loss: %.4f  loss_ce: %.4f  lr: %.6f  elapsed: %.2fh"
                % (epoch_num + 1, max_epoch, avg_loss, avg_loss_ce, lr_, elapsed_h)
            )
            gates.end_epoch(epoch_num + 1, writer)

            save_interval = 50
            if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
                save_mode_path = os.path.join(
                    snapshot_path, "epoch_" + str(epoch_num) + ".pth"
                )
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

        if (
            is_main
            and args.val_interval > 0
            and (epoch_num + 1) % args.val_interval == 0
        ):
            val_t0 = time.time()
            val_dice = _validate_2d(args, model, dataset_class)
            val_time_s += time.time() - val_t0
            logging.info(
                "Validation DSC: %.4f  best: %.4f" % (val_dice, best_performance)
            )
            if val_dice > best_performance:
                best_performance = val_dice
                net = model.module if hasattr(model, "module") else model
                torch.save(
                    net.state_dict(), os.path.join(snapshot_path, "best_model.pth")
                )
                logging.info("=> best model saved (DSC %.4f)" % best_performance)

        if epoch_num >= max_epoch - 1:
            if is_main:
                save_mode_path = os.path.join(
                    snapshot_path, "epoch_" + str(epoch_num) + ".pth"
                )
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    if is_main:
        total_hours = (time.time() - train_start - val_time_s) / 3600
        logging.info(
            "Total training time: {:.2f} hours (excl. {:.1f} min validation overhead)".format(
                total_hours, val_time_s / 60
            )
        )
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            logging.info(
                "Peak VRAM: {:.0f} MB ({:.1f} GB)".format(peak_mb, peak_mb / 1024)
            )
        gates.save(os.path.join(snapshot_path, "gate_history.npz"))
        if writer is not None:
            writer.close()
    if use_ddp:
        dist.destroy_process_group()
    return "Training Finished!"


def trainer_kvasir(args, model, snapshot_path):
    from datasets.dataset_kvasir import Kvasir_dataset, RandomGenerator

    db_train = Kvasir_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="train",
        transform=transforms.Compose([RandomGenerator([args.img_size, args.img_size])]),
        **({"rgb": True} if args.rgb else {}),
    )
    return _trainer_2d(args, model, snapshot_path, db_train, Kvasir_dataset)


def trainer_isic(args, model, snapshot_path):
    from datasets.dataset_isic import ISIC_dataset, RandomGenerator

    db_train = ISIC_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="train",
        transform=transforms.Compose([RandomGenerator([args.img_size, args.img_size], hr_size=256 if args.point_loss else None)]),
        **({"rgb": True} if args.rgb else {}),
    )
    return _trainer_2d(args, model, snapshot_path, db_train, ISIC_dataset)


def trainer_cvc(args, model, snapshot_path):
    from datasets.dataset_cvc import CVC_dataset, RandomGenerator

    db_train = CVC_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="train",
        transform=transforms.Compose([RandomGenerator([args.img_size, args.img_size], hr_size=384 if args.point_loss else None)]),
        **({"rgb": True} if args.rgb else {}),
    )
    return _trainer_2d(args, model, snapshot_path, db_train, CVC_dataset)


def trainer_kvasir_instrument(args, model, snapshot_path):
    from datasets.dataset_kvasir_instrument import (
        KvasirInstrument_dataset,
        RandomGenerator,
    )

    db_train = KvasirInstrument_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="train",
        transform=transforms.Compose([RandomGenerator([args.img_size, args.img_size])]),
        **({"rgb": True} if args.rgb else {}),
    )
    return _trainer_2d(args, model, snapshot_path, db_train, KvasirInstrument_dataset)


def trainer_chest_xray(args, model, snapshot_path):
    from datasets.dataset_chest_xray import ChestXray_dataset, RandomGenerator

    db_train = ChestXray_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="train",
        transform=transforms.Compose([RandomGenerator([args.img_size, args.img_size])]),
        **({"rgb": True} if args.rgb else {}),
    )
    return _trainer_2d(args, model, snapshot_path, db_train, ChestXray_dataset)


def trainer_acdc(args, model, snapshot_path):
    from datasets.dataset_acdc import ACDC_dataset, RandomGenerator

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    use_ddp = dist.is_initialized() and world_size > 1
    is_main = not use_ddp or local_rank == 0

    if is_main:
        logging.basicConfig(
            filename=snapshot_path + "/log.txt",
            level=logging.INFO,
            format="[%(asctime)s.%(msecs)03d] %(message)s",
            datefmt="%H:%M:%S",
        )
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    db_train = ACDC_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="train",
        transform=transforms.Compose(
            [RandomGenerator(output_size=[args.img_size, args.img_size], hr_size=224 if args.point_loss else None)]
        ),
    )
    if is_main:
        print("The length of train set is: {}".format(len(db_train)))
        print(
            "DDP: {} GPU(s), world_size={}".format(
                world_size if use_ddp else 1, world_size
            )
        )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    if use_ddp:
        sampler = DistributedSampler(db_train, shuffle=True)
        trainloader = DataLoader(
            db_train,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=args.drop_last,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    else:
        trainloader = DataLoader(
            db_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=args.drop_last,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )
    model.train()
    epoch_loss_pt = 0.0  # --point_loss
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = build_optimizer(args, model)
    writer = SummaryWriter(snapshot_path + "/log") if is_main else None
    gates = GateMonitor(model)
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    if is_main:
        logging.info(
            "{} iterations per epoch. {} max iterations ".format(
                len(trainloader), max_iterations
            )
        )
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70, disable=not is_main)
    train_start = time.time()
    val_time_s = 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for epoch_num in iterator:
        epoch_loss_sum = 0.0
        epoch_loss_ce_sum = 0.0
        epoch_batches = 0
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            if args.point_loss:
                loss_pt = point_supervision_loss(outputs, sampled_batch["label_hr"].cuda(non_blocking=True),
                                                 class_share=args.ptl_class_share if args.ptl_balanced else 0.0)
                loss = loss + loss_pt
                epoch_loss_pt += loss_pt.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_ * param_group.get("lr_scale", 1.0)
            iter_num = iter_num + 1
            epoch_loss_sum += loss.item()
            epoch_loss_ce_sum += loss_ce.item()
            epoch_batches += 1
            if writer is not None:
                writer.add_scalar("info/lr", lr_, iter_num)
                writer.add_scalar("info/total_loss", loss, iter_num)
                writer.add_scalar("info/loss_ce", loss_ce, iter_num)
                if iter_num % 20 == 0:
                    image = image_batch[0, 0:1, :, :]
                    image = (image - image.min()) / (image.max() - image.min())
                    writer.add_image("train/Image", image, iter_num)
                    outputs_vis = torch.argmax(
                        torch.softmax(outputs, dim=1), dim=1, keepdim=True
                    )
                    writer.add_image(
                        "train/Prediction", outputs_vis[0, ...] * 50, iter_num
                    )
                    labs = label_batch[0, ...].unsqueeze(0) * 50
                    writer.add_image("train/GroundTruth", labs, iter_num)

        if is_main:
            elapsed_h = (time.time() - train_start - val_time_s) / 3600
            avg_loss = epoch_loss_sum / epoch_batches
            avg_loss_ce = epoch_loss_ce_sum / epoch_batches
            if args.point_loss:
                logging.info("epoch %d/%d  loss_point: %.4f" % (epoch_num + 1, max_epoch, epoch_loss_pt / epoch_batches))
                epoch_loss_pt = 0.0
            logging.info(
                "epoch %d/%d  loss: %.4f  loss_ce: %.4f  lr: %.6f  elapsed: %.2fh"
                % (epoch_num + 1, max_epoch, avg_loss, avg_loss_ce, lr_, elapsed_h)
            )
            gates.end_epoch(epoch_num + 1, writer)

            save_interval = 50
            if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
                save_mode_path = os.path.join(
                    snapshot_path, "epoch_" + str(epoch_num) + ".pth"
                )
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            if is_main:
                save_mode_path = os.path.join(
                    snapshot_path, "epoch_" + str(epoch_num) + ".pth"
                )
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    if is_main:
        total_hours = (time.time() - train_start - val_time_s) / 3600
        logging.info(
            "Total training time: {:.2f} hours (excl. {:.1f} min validation overhead)".format(
                total_hours, val_time_s / 60
            )
        )
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            logging.info(
                "Peak VRAM: {:.0f} MB ({:.1f} GB)".format(peak_mb, peak_mb / 1024)
            )
        gates.save(os.path.join(snapshot_path, "gate_history.npz"))
        if writer is not None:
            writer.close()
    if use_ddp:
        dist.destroy_process_group()
    return "Training Finished!"


def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    use_ddp = dist.is_initialized() and world_size > 1
    is_main = not use_ddp or local_rank == 0

    if is_main:
        logging.basicConfig(
            filename=snapshot_path + "/log.txt",
            level=logging.INFO,
            format="[%(asctime)s.%(msecs)03d] %(message)s",
            datefmt="%H:%M:%S",
        )
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    db_train = Synapse_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="train",
        transform=transforms.Compose(
            [RandomGenerator(output_size=[args.img_size, args.img_size], aug=args.aug)]
        ),
    )
    if args.point_refine or args.point_loss:  # also the augmented slice on the native canvas (point supervision)
        hr = 256 if args.dataset == "FeTA" else 512  # FeTA slices are at most 256 x 256 after the foreground crop
        db_train.transform = transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size], aug=args.aug, hr_size=hr)])
    if args.slice_attn >= 0:  # stacks of the 2K+1 adjacent slices, same augmentation per stack
        from datasets.dataset_slices import SliceStack_dataset, StackRandomGenerator

        db_train = SliceStack_dataset(args.root_path, args.list_dir, "train", K=args.slice_attn,
                                      transform=transforms.Compose([StackRandomGenerator([args.img_size, args.img_size], aug=args.aug)]))
    if is_main:
        print("The length of train set is: {}".format(len(db_train)))
        print(
            "DDP: {} GPU(s), world_size={}".format(
                world_size if use_ddp else 1, world_size
            )
        )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    if use_ddp:
        sampler = DistributedSampler(db_train, shuffle=True)
        trainloader = DataLoader(
            db_train,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=args.drop_last,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    else:
        trainloader = DataLoader(
            db_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=args.drop_last,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )
    model.train()
    # --pam_affinity: modules whose attention logits are supervised (Context Prior affinity loss)
    aff_mods = [m for m in model.modules() if getattr(m, "affinity_in", 0)]
    epoch_loss_aff = 0.0
    # --third_branch organ: auxiliary coarse-segmentation loss of the organ branch
    organ_mods = [m for m in model.modules() if hasattr(m, "aux_logits")]
    epoch_loss_org = 0.0
    # --point_refine: point loss of the native-resolution refinement head
    net_ = model.module if hasattr(model, "module") else model
    point_head = getattr(net_, "point_head", None)
    epoch_loss_pt = 0.0
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = build_optimizer(args, model)
    writer = SummaryWriter(snapshot_path + "/log") if is_main else None
    gates = GateMonitor(model)
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(
        trainloader
    )  # max_epoch = max_iterations // len(trainloader) + 1
    if is_main:
        logging.info(
            "{} iterations per epoch. {} max iterations ".format(
                len(trainloader), max_iterations
            )
        )
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70, disable=not is_main)
    train_start = time.time()
    val_time_s = 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for epoch_num in iterator:
        epoch_loss_sum = 0.0
        epoch_loss_ce_sum = 0.0
        epoch_batches = 0
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            if aff_mods:
                loss_aff = sum(affinity_loss(m.affinity_logits, label_batch, num_classes) for m in aff_mods)
                loss = loss + args.pam_affinity * loss_aff
                epoch_loss_aff += loss_aff.item()
            if args.point_loss:
                loss_pt = point_supervision_loss(outputs, sampled_batch["label_hr"].cuda(non_blocking=True),
                                                 class_share=args.ptl_class_share if args.ptl_balanced else 0.0)
                loss = loss + loss_pt
                epoch_loss_pt += loss_pt.item()
            if point_head is not None:
                loss_pt = point_head.train_loss(outputs, net_.last_dec_feat, sampled_batch["image_hr"].cuda(non_blocking=True),
                                                sampled_batch["label_hr"].cuda(non_blocking=True))
                loss = loss + loss_pt
                epoch_loss_pt += loss_pt.item()
            if organ_mods:
                loss_org = sum(organ_aux_loss(m.aux_logits, label_batch, num_classes) for m in organ_mods)
                loss = loss + args.organ_aux * loss_org
                epoch_loss_org += loss_org.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_ * param_group.get("lr_scale", 1.0)

            iter_num = iter_num + 1
            epoch_loss_sum += loss.item()
            epoch_loss_ce_sum += loss_ce.item()
            epoch_batches += 1
            if writer is not None:
                writer.add_scalar("info/lr", lr_, iter_num)
                writer.add_scalar("info/total_loss", loss, iter_num)
                writer.add_scalar("info/loss_ce", loss_ce, iter_num)
                if iter_num % 20 == 0:
                    image = image_batch[1, 0:1, :, :]
                    image = (image - image.min()) / (image.max() - image.min())
                    writer.add_image("train/Image", image, iter_num)
                    outputs_vis = torch.argmax(
                        torch.softmax(outputs, dim=1), dim=1, keepdim=True
                    )
                    writer.add_image(
                        "train/Prediction", outputs_vis[1, ...] * 50, iter_num
                    )
                    labs = label_batch[1, ...].unsqueeze(0) * 50
                    writer.add_image("train/GroundTruth", labs, iter_num)

        if is_main:
            elapsed_h = (time.time() - train_start - val_time_s) / 3600
            avg_loss = epoch_loss_sum / epoch_batches
            avg_loss_ce = epoch_loss_ce_sum / epoch_batches
            logging.info(
                "epoch %d/%d  loss: %.4f  loss_ce: %.4f  lr: %.6f  elapsed: %.2fh"
                % (epoch_num + 1, max_epoch, avg_loss, avg_loss_ce, lr_, elapsed_h)
            )
            if aff_mods:
                logging.info("epoch %d/%d  loss_aff: %.4f" % (epoch_num + 1, max_epoch, epoch_loss_aff / epoch_batches))
                epoch_loss_aff = 0.0
            if point_head is not None or args.point_loss:
                logging.info("epoch %d/%d  loss_point: %.4f" % (epoch_num + 1, max_epoch, epoch_loss_pt / epoch_batches))
                epoch_loss_pt = 0.0
            if organ_mods:
                logging.info("epoch %d/%d  loss_organ_aux: %.4f" % (epoch_num + 1, max_epoch, epoch_loss_org / epoch_batches))
                epoch_loss_org = 0.0
            gates.end_epoch(epoch_num + 1, writer)

            save_interval = 50  # int(max_epoch/6)
            if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
                save_mode_path = os.path.join(
                    snapshot_path, "epoch_" + str(epoch_num) + ".pth"
                )
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

        should_val = args.val_interval > 0 and (
            (epoch_num + 1) % args.val_interval == 0 or epoch_num >= max_epoch - 1
        )
        if should_val:
            # All ranks validate in parallel on their own GPU — no barriers needed.
            # Validation uses net (model.module), bypassing DDP, so no NCCL ops fire.
            # Ranks naturally re-sync at the next epoch's loss.backward() all-reduce.
            _val_t0 = time.time()
            val_dice = _validate_synapse(args, model)
            val_time_s += time.time() - _val_t0
            if is_main:
                logging.info(
                    "epoch %d val_dice: %.4f  best: %.4f"
                    % (epoch_num + 1, val_dice, best_performance)
                )
                if val_dice > best_performance:
                    best_performance = val_dice
                    net = model.module if hasattr(model, "module") else model
                    torch.save(
                        net.state_dict(), os.path.join(snapshot_path, "best_model.pth")
                    )
                    logging.info("=> best model saved (DSC %.4f)" % best_performance)

        if epoch_num >= max_epoch - 1:
            if is_main:
                save_mode_path = os.path.join(
                    snapshot_path, "epoch_" + str(epoch_num) + ".pth"
                )
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    if is_main:
        total_hours = (time.time() - train_start - val_time_s) / 3600
        logging.info(
            "Total training time: {:.2f} hours (excl. {:.1f} min validation overhead)".format(
                total_hours, val_time_s / 60
            )
        )
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            logging.info(
                "Peak VRAM: {:.0f} MB ({:.1f} GB)".format(peak_mb, peak_mb / 1024)
            )
        gates.save(os.path.join(snapshot_path, "gate_history.npz"))
        if writer is not None:
            writer.close()
    if use_ddp:
        dist.destroy_process_group()
    return "Training Finished!"


if __name__ == "__main__":
    # Python 3.14 defaults to forkserver on Linux, which cannot pickle the local
    # worker_init_fn; keep the fork start method the original runs used.
    torch.multiprocessing.set_start_method("fork")
    # DDP setup: torchrun injects LOCAL_RANK; plain `python train.py` leaves it unset (-1).
    # No --n_gpu check needed — WORLD_SIZE from the environment is the authoritative source.
    _local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if _local_rank >= 0:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(_local_rank)

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
    dataset_name = args.dataset
    dataset_config = {
        "Synapse": {
            "root_path": "../data/Synapse/train_npz",
            "list_dir": "./lists/lists_Synapse",
            "num_classes": 9,
        },
        "AMOS": {
            "root_path": "../data/AMOS_CT_2d/train_npz",
            "list_dir": "./lists/lists_AMOS",
            "num_classes": 16,
        },
        "FeTA": {
            "root_path": "../data/FeTA_2d/train_npz",
            "list_dir": "./lists/lists_FeTA",
            "num_classes": 8,
        },
        "ACDC": {
            "root_path": "../data/ACDC_MTUNet",
            "list_dir": "./lists/lists_ACDC",
            "num_classes": 4,
        },
        "Kvasir": {
            "root_path": "../data/Kvasir-SEG",
            "list_dir": "./lists/lists_Kvasir",
            "num_classes": 2,
        },
        "ISIC": {
            "root_path": "../data/ISIC2018",
            "list_dir": "./lists/lists_ISIC",
            "num_classes": 2,
        },
        "CVC": {
            "root_path": "../data/CVC-ClinicDB",
            "list_dir": "./lists/lists_CVC",
            "num_classes": 2,
        },
        "KvasirInstrument": {
            "root_path": "../data/Kvasir-Instrument",
            "list_dir": "./lists/lists_KvasirInstrument",
            "num_classes": 2,
        },
        "ChestXray": {
            "root_path": "../data/Montgomery",
            "list_dir": "./lists/lists_ChestXray",
            "num_classes": 2,
        },
    }
    #     if args.batch_size != 24 and args.batch_size % 6 == 0:
    #         args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]["num_classes"]
    args.root_path = dataset_config[dataset_name]["root_path"]
    args.list_dir = dataset_config[dataset_name]["list_dir"]
    args.is_pretrain = True
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
        snapshot_path + "_" + str(args.max_iterations)[0:2] + "k"
        if args.max_iterations != 30000
        else snapshot_path
    )
    snapshot_path = (
        snapshot_path + "_epo" + str(args.max_epochs)
        if args.max_epochs != 30
        else snapshot_path
    )
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
    if args.point_refine and (args.dataset not in ("Synapse", "AMOS") or args.slice_attn >= 0):
        raise SystemExit("--point_refine is implemented in trainer_synapse (Synapse / AMOS), without --slice_attn")
    if args.point_loss and (args.dataset not in ("Synapse", "AMOS", "FeTA", "ACDC", "CVC", "ISIC") or args.slice_attn >= 0):
        raise SystemExit("--point_loss is implemented for Synapse, AMOS, FeTA, ACDC, CVC and ISIC (without --slice_attn)")
    if args.slice_attn >= 0 and args.dataset not in ("Synapse", "AMOS"):
        raise SystemExit("--slice_attn is implemented in trainer_synapse (Synapse / AMOS) only")
    if args.third_branch != "none" and args.dataset not in ("Synapse", "AMOS"):
        raise SystemExit("--third_branch is implemented in trainer_synapse (Synapse / AMOS) only")
    if args.pam_affinity > 0 and args.dataset not in ("Synapse", "AMOS"):
        raise SystemExit("--pam_affinity is implemented in trainer_synapse (Synapse / AMOS) only")
    if args.aug != "default" and args.dataset not in ("Synapse", "AMOS"):
        raise SystemExit("--aug is implemented for Synapse and AMOS only")
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
    if args.rgb and dataset_name not in ("Kvasir", "ISIC", "CVC"):
        raise SystemExit("--rgb is only implemented for Kvasir, ISIC and CVC")
    check_val_split(args)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
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
    if args.vit_name.find("R50") != -1:
        config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size),
            int(args.img_size / args.vit_patches_size),
        )
    net = ApproxDATransUNet(
        config_vit, img_size=args.img_size, num_classes=config_vit.n_classes
    ).cuda()
    net.load_from(weights=np.load(config_vit.pretrained_path))

    trainer = {
        "Synapse": trainer_synapse,
        "AMOS": trainer_synapse,  # same slice format (datasets/prepare_amos.py)
        "FeTA": trainer_synapse,  # same slice format (datasets/prepare_feta.py)
        "ACDC": trainer_acdc,
        "Kvasir": trainer_kvasir,
        "ISIC": trainer_isic,
        "CVC": trainer_cvc,
        "KvasirInstrument": trainer_kvasir_instrument,
        "ChestXray": trainer_chest_xray,
    }
    trainer[dataset_name](args, net, snapshot_path)
