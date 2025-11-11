"""Evaluate trained checkpoints on the held-out final seeds, across training seeds.

For each variant and training seed, loads ckpts/<variant>_seed<s>/<tag>.pt, runs the
policy on cfg.final_eval_seeds, and writes results/metrics/eval_<variant>.json with
per-seed results and across-seed mean and sample SD.

Example:
    python scripts/evaluate.py --variants sac vcara_no_cf vcara --seeds 1 2 3 4 --ckpt-root outputs
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vcara import VCARAAgent, make_cfg, evaluate  # noqa: E402

METRICS = ("mean_return", "std_return", "completion", "out_of_field", "mean_coverage",
           "median_coverage", "p90_coverage", "off_road_frac", "mean_speed", "t90")


def aggregate(per_seed):
    """Across-seed mean and sample SD for each metric. t90 ignores seeds where it is undefined (-1)."""
    out = {}
    for k in METRICS:
        vals = np.array([r[k] for r in per_seed if k in r], dtype=float)
        if k == "t90":
            vals = vals[vals >= 0]
        if len(vals) == 0:
            continue
        out[k] = dict(mean=float(vals.mean()),
                      sd_across_seeds=float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
                      n=int(len(vals)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["vcara"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--ckpt-root", default="outputs")
    ap.add_argument("--tag", default="final", help="checkpoint name inside each seed folder")
    ap.add_argument("--stochastic", action="store_true", help="sample actions instead of using the mean")
    ap.add_argument("--out", default="results/metrics")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    for v in args.variants:
        cfg = make_cfg(v)
        per_seed, ck_paths = [], []
        for s in args.seeds:
            ck = os.path.join(args.ckpt_root, "ckpts", f"{v}_seed{s}", f"{args.tag}.pt")
            if not os.path.exists(ck):
                print(f"[{v} s{s}] missing {ck}; skipped")
                continue
            agent = VCARAAgent(cfg, dev)
            meta = agent.load_policy(ck)
            res = evaluate(agent, cfg, list(cfg.final_eval_seeds), deterministic=not args.stochastic)
            per_seed.append(dict(seed=s, frames_total=int(meta.get("frames_total", -1)), **res))
            ck_paths.append(ck)
            print(f"[{v} s{s}] return={res['mean_return']:.1f} cov={res['mean_coverage']:.3f} "
                  f"off={res['off_road_frac']:.3f} lap={res['completion']:.2f}")

        if not per_seed:
            print(f"[{v}] no checkpoints found; nothing written")
            continue
        out = dict(variant=v, tag=args.tag, stochastic=args.stochastic,
                   eval_seeds=list(cfg.final_eval_seeds), n_seeds=len(per_seed),
                   per_seed=per_seed, aggregate=aggregate(per_seed), ckpt_paths=ck_paths)
        path = os.path.join(args.out, f"eval_{v}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        r = out["aggregate"]["mean_return"]
        print(f"[{v}] {len(per_seed)} seeds: return {r['mean']:.1f} +- {r['sd_across_seeds']:.1f} "
              f"(across-seed SD) -> {path}")


if __name__ == "__main__":
    main()
