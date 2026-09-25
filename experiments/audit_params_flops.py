"""Parameter / FLOP audit: DA-TransUNet vs ApproxDA-TransUNet.

Counts parameters statically and FLOPs with fvcore (same library and input as
test.py: one 1x224x224 slice, fvcore counts one multiply-add as one FLOP and
includes attention bmm). Every number is broken down by component, separating
attention blocks that actually run in forward() from ones that are built but
never called.

Usage (from experiments/):
    python audit_params_flops.py
"""

import importlib
import os
import sys
import types
import warnings
from collections import defaultdict

import torch

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))

try:
    import ml_collections  # noqa: F401
except ImportError:  # configs.py only needs attribute-style dicts

    class _ConfigDict(dict):
        def __getattr__(self, k):
            try:
                return self[k]
            except KeyError:
                raise AttributeError(k)

        def __setattr__(self, k, v):
            self[k] = v

    sys.modules["ml_collections"] = types.SimpleNamespace(ConfigDict=_ConfigDict)

from fvcore.nn import FlopCountAnalysis  # noqa: E402


def load_arch(project, module, cls):
    """Import <project>/Architecture/<module>.py, isolating the two projects."""
    for name in list(sys.modules):
        if name == "Architecture" or name.startswith("Architecture."):
            del sys.modules[name]
    sys.path.insert(0, os.path.join(HERE, project))
    try:
        mod = importlib.import_module("Architecture." + module)
        return getattr(mod, cls), mod.CONFIGS
    finally:
        sys.path.pop(0)


def build_da():
    Net, CONFIGS = load_arch("DA-TransUNet", "DATransUNet", "DA_Transformer")
    cfg = CONFIGS["R50-ViT-B_16"]
    cfg.n_classes, cfg.n_skip, cfg.patches.grid = 9, 3, (14, 14)
    return Net(cfg, img_size=224, num_classes=9).eval(), "DANetHead"


def build_approx(M, r=32, G=8, gate="pam", version="c16"):
    Net, CONFIGS = load_arch("ApproxDA-TransUNet", "ApproxDATransUNet", "ApproxDATransUNet")
    cfg = CONFIGS["R50-ViT-B_16"]
    cfg.n_classes, cfg.n_skip, cfg.patches.grid = 9, 3, (14, 14)
    cfg.window_size, cfg.rank, cfg.groups, cfg.gate_mode = M, r, G, gate
    cfg.block_version = version
    cls = {"c16": "ApproxDABlock", "legacy": "ApproxDABlockLegacy"}[version]
    return Net(cfg, img_size=224, num_classes=9).eval(), cls


def group_of(name, attn_blocks):
    """Map a parameter / module name to a report group."""
    for blk in attn_blocks:
        if name == blk or name.startswith(blk + "."):
            return blk
    if name.startswith("transformer.embeddings.hybrid_model"):
        return "CNN backbone (R50)"
    if name.startswith("transformer.embeddings"):
        return "ViT patch + position embedding"
    if name.startswith("transformer.encoder"):
        return "ViT encoder (12 layers)"
    if name.startswith("decoder"):
        return "Decoder convs"
    if name.startswith("segmentation_head"):
        return "Segmentation head"
    return "Other"


def audit(label, net, block_cls):
    attn_blocks = [n for n, m in net.named_modules() if type(m).__name__ == block_cls]
    called = {}

    def make_hook(name):
        def hook(mod, inputs, output):  # must return None, or it replaces output
            called.setdefault(name, tuple(inputs[0].shape[1:]))

        return hook

    hooks = [
        m.register_forward_hook(make_hook(n))
        for n, m in net.named_modules()
        if n in attn_blocks
    ]
    x = torch.randn(1, 1, 224, 224)
    with torch.no_grad():
        net(x)
    for h in hooks:
        h.remove()

    # --- parameters (static) ---
    p_group = defaultdict(int)
    for n, p in net.named_parameters():
        p_group[group_of(n, attn_blocks)] += p.numel()

    # --- FLOPs (fvcore, one multiply-add = 1 FLOP) ---
    fa = FlopCountAnalysis(net, x)
    fa.unsupported_ops_warnings(False)
    fa.uncalled_modules_warnings(False)
    by_mod = fa.by_module()
    f_group = defaultdict(int)
    # Attribute each module's *own* ops (its total minus its direct children),
    # so ops issued inside non-leaf modules (e.g. attention matmuls) are counted.
    for n, m in net.named_modules():
        if not n:
            continue
        children = sum(by_mod.get(f"{n}.{c}", 0) for c, _ in m.named_children())
        f_group[group_of(n, attn_blocks)] += by_mod.get(n, 0) - children
    total_p = sum(p_group.values())
    total_f = fa.total()

    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    print(f"{'component':44s}{'params (M)':>12s}{'%':>6s}{'GFLOPs':>10s}{'%':>6s}")
    rows = [g for g in p_group if g not in attn_blocks]
    for g in sorted(rows, key=lambda g: -p_group[g]):
        print(f"{g:44s}{p_group[g] / 1e6:12.2f}{100 * p_group[g] / total_p:6.1f}"
              f"{f_group[g] / 1e9:10.2f}{100 * f_group[g] / total_f:6.1f}")
    used = [b for b in attn_blocks if b in called]
    unused = [b for b in attn_blocks if b not in called]
    for b in used:
        tag = f"  attn {b} {called[b]}"
        print(f"{tag:44s}{p_group[b] / 1e6:12.2f}{100 * p_group[b] / total_p:6.1f}"
              f"{f_group[b] / 1e9:10.2f}{100 * f_group[b] / total_f:6.1f}")
    pu = sum(p_group[b] for b in unused)
    print(f"{'  attn blocks NEVER CALLED (x' + str(len(unused)) + ')':44s}"
          f"{pu / 1e6:12.2f}{100 * pu / total_p:6.1f}{0:10.2f}{0:6.1f}")
    pa = sum(p_group[b] for b in used)
    fa_used = sum(f_group[b] for b in used)
    print(f"{'-' * 78}\n{'TOTAL':44s}{total_p / 1e6:12.2f}{'':6s}{total_f / 1e9:10.2f}")
    print(f"{'  of which attention blocks that run':44s}{pa / 1e6:12.2f}"
          f"{100 * pa / total_p:6.1f}{fa_used / 1e9:10.2f}{100 * fa_used / total_f:6.1f}")
    print(f"{'  total excluding never-called blocks':44s}{(total_p - pu) / 1e6:12.2f}")
    return net, used, by_mod


def block_detail(label, net, used, by_mod):
    """Per-submodule params / GFLOPs inside each attention block that runs."""
    print(f"\n--- {label}: inside each attention block that runs ---")
    for blk in used:
        mod = dict(net.named_modules())[blk]
        print(f"{blk}")
        for sub, m in mod.named_children():
            full = f"{blk}.{sub}"
            p = sum(q.numel() for q in m.parameters())
            f = by_mod.get(full, 0)
            print(f"    {sub:12s}{type(m).__name__:22s}params {p / 1e6:7.3f}M   GFLOPs {f / 1e9:7.3f}")
        own = by_mod.get(blk, 0) - sum(by_mod.get(f"{blk}.{s}", 0) for s, _ in mod.named_children())
        if own:
            print(f"    {'(own ops)':34s}{'':16s}GFLOPs {own / 1e9:7.3f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    da, da_cls = build_da()
    r = audit("DA-TransUNet (full dual attention, C/16 bottleneck)", da, da_cls)
    block_detail("DA-TransUNet", *r)
    for version in ("legacy", "c16"):
        for M, r_, gate in [(7, 32, "pam"), (28, 32, "pam"), (112, 32, "pam"),
                            (112, 0, "pam"), (7, 32, "learn")]:
            net, cls = build_approx(M, r=r_, gate=gate, version=version)
            res = audit(f"ApproxDA-TransUNet [{version}] (M={M}, r={r_ or 'none'}, G=8, gate={gate})", net, cls)
            if (version, M, r_, gate) == ("c16", 28, 32, "pam"):
                block_detail(f"ApproxDA c16 M={M}", *res)
