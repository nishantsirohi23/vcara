"""Record evaluation rollouts from a saved checkpoint to MP4.

Example:
    python scripts/record_videos.py --ckpt outputs/vcara_seed42/ckpts/vcara_seed42/best_eval_cov.pt \
        --variant vcara --seeds 2000 2001 2002 --out videos/full
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vcara import VCARAAgent, make_cfg, evaluate  # noqa: E402


def write_mp4(frames, path, fps=50):
    """frames: list of RGB uint8 arrays (H, W, 3)."""
    if not frames:
        raise ValueError("no frames recorded")
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(np.asarray(f, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", default="vcara")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2000])
    ap.add_argument("--stochastic", action="store_true", help="sample actions instead of the mean")
    ap.add_argument("--out", default="videos/full")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = make_cfg(args.variant)
    agent = VCARAAgent(cfg, dev)
    agent.load_policy(args.ckpt)
    os.makedirs(args.out, exist_ok=True)
    tag = os.path.splitext(os.path.basename(args.ckpt))[0]

    for sd in args.seeds:
        stats, frames = evaluate(agent, cfg, [sd], deterministic=not args.stochastic, record=True)
        stem = f"{args.variant}_{tag}_seed{sd}"
        write_mp4(frames, os.path.join(args.out, f"{stem}.mp4"))
        with open(os.path.join(args.out, f"{stem}.json"), "w") as f:
            json.dump(dict(ckpt=args.ckpt, seed=sd, stochastic=args.stochastic, **stats), f, indent=2)
        print(f"wrote {stem}.mp4  coverage={stats['mean_coverage']:.3f} "
              f"return={stats['mean_return']:.1f} lap={stats['completion']:.0f}")


if __name__ == "__main__":
    main()
