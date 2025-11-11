"""Train one V-CARA variant with one seed.

Examples:
    python scripts/train.py --variant vcara --seed 42 --frames 1000000 --out outputs
    python scripts/train.py --variant vcara --smoke --out outputs
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vcara import VARIANTS, make_cfg, train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="vcara", choices=list(VARIANTS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frames", type=int, default=2_000_000)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end run to check the pipeline")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70, f"\nV-CARA variant={args.variant} seed={args.seed} device={dev}\n" + "=" * 70)
    if args.smoke:
        cfg = make_cfg(args.variant, total_frames=4000, start_steps=150, batch=32, buffer=4000,
                       eval_every=2000, eval_episodes=1, eval_stoch_episodes=1, cf_warmup=40, cf_ramp=40,
                       log_every=50, ckpt_every=3000, near_finish_threshold=0.0, elite_threshold=0.0,
                       elite_offroad_max=1.0, finish_label_cov=0.0, final_eval_seeds=(2000,))
        out = os.path.join(args.out, "smoke")
    else:
        cfg = make_cfg(args.variant, total_frames=args.frames)
        out = args.out
    train(cfg, name=args.variant, seed=args.seed, out_dir=out, device=dev, verbose=True)
    print("DONE")


if __name__ == "__main__":
    main()
