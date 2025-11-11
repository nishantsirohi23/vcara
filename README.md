# V-CARA: Uncertainty-Gated Counterfactual Reinforcement Learning for Visual Control

Code, configurations, evaluation summaries and videos for **V-CARA**, a visual
reinforcement-learning agent built on Soft Actor-Critic (SAC) for CarRacing-v3
(continuous control, 64x64 grayscale frames, 4-frame stack).

V-CARA adds to SAC: a latent transition ensemble, uncertainty-gated local
counterfactual action evaluation, risk and progress heads, a future-coverage
classifier, and coverage-aware replay.

## Status of results

| Item | What is available |
|---|---|
| Full V-CARA return | 4 independent training seeds (A-D), checkpoints 100k-1M frames |
| Full V-CARA coverage / off-road / speed / T90 | Aggregate trajectory only; not yet available per seed |
| No-CF ablation | 1 training seed, checkpoints 25k-450,002 frames |
| Raw per-episode evaluation files | **Not yet included** (see `data/README.md`) |

Headline numbers at 1M frames (aggregate summary in the paper):
return 875.7 +/- 30.0 (within-evaluation SD), coverage 0.985, off-road 0.090, T90 650.
Four-seed return mean at 1M: 875.75, across-seed sample SD 3.30.

Comparisons with no-CF are **descriptive**: one no-CF seed, no paired test.
See the paper for the full discussion and limitations.

## Repository layout

| Path | Contents |
|---|---|
| `src/vcara/` | Algorithm, networks, replay, environment wrapper, evaluation |
| `scripts/` | Training, video recording, figure regeneration |
| `configs/` | Run configurations (full V-CARA, no-CF) |
| `experiments/` | Experiment launch scripts and seed lists |
| `tests/` | Unit tests and configuration checks |
| `data/` | Evaluation summary schemas and documented data |
| `results/` | Metrics, tables and generated figures |
| `videos/` | Recorded evaluation rollouts (`full/`, `no_cf/`) |
| `notebooks/` | Analysis and demonstration notebooks |
| `paper/` | Manuscript PDF, LaTeX source, bibliography |
| `.github/workflows/` | CI: tests and lint |

## Installation

Requires Python 3.10+. A CUDA GPU is strongly recommended.

```bash
git clone https://github.com/vcara/vcara.git
cd vcara
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

CarRacing-v3 requires Box2D, which is installed through `gymnasium[box2d]`.

## Quick start

Smoke test (tiny run that checks the pipeline end to end):

```bash
python scripts/train.py --variant vcara --smoke --out outputs
```

Full V-CARA, one seed:

```bash
python scripts/train.py --variant vcara --seed 42 --frames 1000000 --out outputs
```

No-counterfactual ablation:

```bash
python scripts/train.py --variant vcara_no_cf --seed 42 --frames 1000000 --out outputs
```

Record evaluation videos from a saved policy:

```bash
python scripts/record_videos.py --ckpt outputs/vcara_seed42/ckpts/vcara_seed42/best_eval_cov.pt \
    --variant vcara --out videos/full
```

Run the tests:

```bash
pytest
```

## Variants

Defined in `src/vcara/core.py` (`VARIANTS`). Baselines share the same environment
fixes (brake mapping, time-aware input) so comparisons are fair.

- Baselines: `sac`, `sac_n3`, `sac_n5`
- Full system: `vcara`, `vcara_stall_trunc`
- Ablations: `vcara_no_cf`, `vcara_no_elite`, `vcara_no_progress`, `vcara_no_finish`,
  `vcara_no_risk`, `vcara_no_time`, `vcara_gru`, `vcara_adaptive`, `vcara_aggressive`

## Known gaps

- The paper's description of the counterfactual score weights and candidate
  offsets is incomplete; the values used by the code are in `configs/` and
  `src/vcara/core.py`. Confirm they match the paper before citing them.
- The paper's numbers have not yet been regenerated from raw per-episode logs
  in this repository. `results/` will hold regenerated tables once those logs are added.
- Training is single-GPU and not bit-reproducible across hardware.

## Citation

See `CITATION.cff`. Paper: N. Sirohi, *V-CARA: Uncertainty-Gated Counterfactual
Reinforcement Learning for Visual Control on an Infinite Track* (2026).

## License

MIT. See `LICENSE`.
