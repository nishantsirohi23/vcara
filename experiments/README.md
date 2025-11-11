Experiment definitions.

| Experiment | Config | Seeds | Status |
|---|---|---|---|
| Full V-CARA return (paper Sec. 6.1) | `configs/vcara_full.yaml` | four seeds, labelled A-D in the paper | Seed values to be added to the config |
| No-CF ablation (paper Sec. 6.2-6.3) | `configs/no_cf.yaml` | one seed (42) | Complete through 450,002 frames |

Launch scripts: `run_full.sh` and `run_no_cf.sh`. Both accept a seed as the first argument.
