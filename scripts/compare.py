"""Build a comparison table from results/metrics/eval_<variant>.json files.

Example:
    python scripts/compare.py --out results/tables/comparison.csv
"""
import argparse
import csv
import glob
import json
import math
import os


def fmt(agg, key, digits=2):
    a = agg.get(key)
    if a is None:
        return "n/a"
    sd = a["sd_across_seeds"]
    sd_s = "n/a" if (sd is None or (isinstance(sd, float) and math.isnan(sd))) else f"{sd:.{digits}f}"
    return f"{a['mean']:.{digits}f} +- {sd_s}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=None,
                    help="eval JSON files (default: results/metrics/eval_*.json)")
    ap.add_argument("--out", default="results/tables/comparison.csv")
    args = ap.parse_args()

    files = args.files or sorted(glob.glob("results/metrics/eval_*.json"))
    if not files:
        raise SystemExit("no eval JSON files found; run scripts/evaluate.py first")

    cols = ["mean_return", "mean_coverage", "off_road_frac", "completion", "t90"]
    header = ["variant", "n_seeds"] + cols
    rows = []
    for p in files:
        d = json.load(open(p))
        agg = d["aggregate"]
        rows.append([d["variant"], d["n_seeds"]] + [fmt(agg, c, 3 if c in ("mean_coverage", "off_road_frac", "completion") else 1)
                                                     for c in cols])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")
    print(f"\nwrote {args.out}  (values are across-seed mean +- sample SD)")


if __name__ == "__main__":
    main()
