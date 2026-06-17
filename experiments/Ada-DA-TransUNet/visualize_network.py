"""
Network visualization for ApproxDA-TransUNet using torchview.

Usage:
    python visualize_network.py                     # renders all 3 presets
    python visualize_network.py --preset overview   # compact architecture figure
    python visualize_network.py --preset block      # ApproxDABlock internals
    python visualize_network.py --preset full       # full expansion (large)
    python visualize_network.py --gate_mode pam --n_classes 9
    python visualize_network.py --format png        # output format (pdf, png, svg)
    python visualize_network.py --out_dir ./graphs

Presets
-------
  overview  depth=2, expand_nested=False  — high-level block diagram for paper figure
  block     depth=4, expand_nested=False  — shows inside ApproxDABlock
  full      depth=6, expand_nested=True   — full graph (large; use for debugging)
"""
import sys
import os
import argparse

import torch

# Allow running from the experiments/Ada-DA-TransUNet directory or from repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from Architecture.AdaDATransUNet import AdaDATransUNet, CONFIGS


PRESETS = {
    "overview": dict(depth=2, expand_nested=False),
    "block":    dict(depth=4, expand_nested=False),
    "full":     dict(depth=6, expand_nested=True),
}


def build_model(gate_mode="pam", n_classes=9, window_size=7, rank=32, groups=8):
    config = CONFIGS["R50-ViT-B_16"]
    config.n_classes = n_classes
    config.n_skip = 3
    config.patches.grid = (14, 14)
    config.window_size = window_size
    config.rank = rank
    config.groups = groups
    config.gate_mode = gate_mode
    # Skip pretrained weight loading — not needed for architecture visualization
    config.pretrained_path = None
    config.resnet_pretrained_path = None

    model = AdaDATransUNet(config, img_size=224, num_classes=n_classes)
    model.eval()
    return model


def render_graph(model, dummy_input, preset_name, preset_cfg, out_dir, fmt):
    try:
        from torchview import draw_graph
    except ImportError:
        print("torchview not installed. Run: pip install torchview")
        sys.exit(1)

    graph_name = f"ApproxDA-TransUNet ({preset_name})"
    filename = f"approxda_{preset_name}"

    print(f"  [{preset_name}] depth={preset_cfg['depth']}, "
          f"expand_nested={preset_cfg['expand_nested']} ... ", end="", flush=True)

    model_graph = draw_graph(
        model,
        input_data=dummy_input,
        graph_name=graph_name,
        **preset_cfg,
    )

    out_path = os.path.join(out_dir, filename)
    # render() saves <out_path>.<fmt> and removes the intermediate .gv source
    model_graph.visual_graph.render(
        filename=out_path,
        format=fmt,
        cleanup=True,
    )
    print(f"saved → {out_path}.{fmt}")
    return model_graph


def main():
    parser = argparse.ArgumentParser(description="Render ApproxDA-TransUNet architecture graph")
    parser.add_argument("--preset", choices=list(PRESETS) + ["all"], default="all",
                        help="Rendering preset (default: all)")
    parser.add_argument("--gate_mode", default="pam",
                        choices=["learn", "fixed", "pam", "cam"],
                        help="Gate mode to visualize (default: pam)")
    parser.add_argument("--n_classes", type=int, default=9,
                        help="Number of output classes (default: 9 for Synapse)")
    parser.add_argument("--window_size", type=int, default=7)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--format", dest="fmt", default="pdf",
                        choices=["pdf", "png", "svg"],
                        help="Output format (default: pdf)")
    parser.add_argument("--out_dir", default="./network_graphs",
                        help="Output directory (default: ./network_graphs)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Building model: gate_mode={args.gate_mode}, n_classes={args.n_classes}, "
          f"window_size={args.window_size}, rank={args.rank}")
    model = build_model(
        gate_mode=args.gate_mode,
        n_classes=args.n_classes,
        window_size=args.window_size,
        rank=args.rank,
        groups=args.groups,
    )
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Single-channel grayscale input (CT / medical imaging)
    dummy_input = torch.randn(1, 1, 224, 224)

    presets_to_run = PRESETS if args.preset == "all" else {args.preset: PRESETS[args.preset]}

    print(f"Rendering {len(presets_to_run)} graph(s) → {args.out_dir}/ [{args.fmt}]")
    for name, cfg in presets_to_run.items():
        render_graph(model, dummy_input, name, cfg, args.out_dir, args.fmt)

    print("Done.")


if __name__ == "__main__":
    main()
