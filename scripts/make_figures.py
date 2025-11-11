"""Regenerate Table 2-style summaries and trajectory figures from a run's JSON log.

Example:
    python scripts/make_figures.py --log outputs/vcara_seed42.json --out results
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, nargs="+", help="one or more run JSON logs")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "tables"), exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    for path in args.log:
        run = json.load(open(path))
        ev = [e for e in run["evals"] if not e.get("final")]
        x = [e["frames"] / 1000 for e in ev]
        ax.errorbar(x, [e["mean_return"] for e in ev], yerr=[e["std_return"] for e in ev],
                    marker="o", ms=3, capsize=2, label=f"{run['name']} s{run['seed']}")
    ax.set_xlabel("Environment frames (thousands)"); ax.set_ylabel("Return")
    ax.set_title("Evaluation return over training (error bars: within-evaluation SD)")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(args.out, "figures", "return_trajectory.png"), dpi=150)

    rows = ["frames,mean_return,std_return,mean_coverage,median_coverage,p90_coverage,off_road_frac,mean_speed,t90"]
    for path in args.log:
        run = json.load(open(path))
        for e in run["evals"]:
            rows.append(",".join(str(e.get(k, "")) for k in
                                 ("frames", "mean_return", "std_return", "mean_coverage", "median_coverage",
                                  "p90_coverage", "off_road_frac", "mean_speed", "t90")))
    with open(os.path.join(args.out, "tables", "checkpoints.csv"), "w") as f:
        f.write("\n".join(rows) + "\n")
    print("wrote results/figures/return_trajectory.png and results/tables/checkpoints.csv")


if __name__ == "__main__":
    main()
