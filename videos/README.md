# Videos

Recorded evaluation rollouts, one MP4 per seed, each with a JSON file of its episode statistics.

| Folder | Contents |
|---|---|
| `full/` | Full V-CARA rollouts |
| `no_cf/` | No-counterfactual ablation rollouts |

Record with:

```bash
python scripts/record_videos.py --ckpt <path/to/checkpoint.pt> --variant vcara --seeds 2000 2001 --out videos/full
```

Naming: `<variant>_<checkpoint>_seed<N>.mp4` and matching `.json`.

Full videos are large. Store them with Git LFS (`git lfs track "videos/**/*.mp4"`),
and commit only a short `preview_*.mp4` per folder to keep the clone light.
Seeds 2000-2019 are the final evaluation seeds set in `Cfg.final_eval_seeds`.

Each video shows one evaluation episode run with the deterministic policy by default.
Stochastic rollouts (`--stochastic`) are recorded separately, and training-time
stall cuts are not shown.
