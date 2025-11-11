"""Configuration checks: the code, the YAML files and the paper's stated values must agree."""
import os

import yaml

from vcara import VARIANTS, make_cfg

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_candidate_index_zero_is_nominal():
    cfg = make_cfg("vcara")
    assert tuple(cfg.cand_deltas[0]) == (0, 0, 0)


def test_paper_reported_values():
    cfg = make_cfg("vcara")
    assert cfg.gamma == 0.99
    assert cfg.tau == 0.005
    assert cfg.actor_lr == 1e-4
    assert cfg.critic_lr == 2e-4
    assert cfg.batch == 256
    assert cfg.buffer == 300_000
    assert cfg.ensemble == 3
    assert cfg.fixed_horizon == 3
    assert cfg.cf_uncertainty_limit == 1.5
    assert cfg.cf_margin == 0.02
    assert cfg.cf_target_rate == 0.25
    assert cfg.cf_warmup == 10_000
    assert cfg.stack == 4 and cfg.img == 64
    assert cfg.action_repeat == 4


def test_replay_mixture_sums_to_one():
    cfg = make_cfg("vcara")
    assert abs(cfg.normal_ratio + cfg.near_finish_ratio + cfg.elite_ratio - 1.0) < 1e-9


def test_no_cf_variant_only_disables_cf():
    full, nocf = make_cfg("vcara"), make_cfg("vcara_no_cf")
    assert nocf.use_cf is False
    assert full.use_risk == nocf.use_risk and full.use_finish == nocf.use_finish


def test_yaml_matches_code():
    for name, fname in (("vcara", "vcara_full.yaml"), ("vcara_no_cf", "no_cf.yaml")):
        with open(os.path.join(ROOT, "configs", fname)) as f:
            y = yaml.safe_load(f)
        assert y["variant"] == name
    with open(os.path.join(ROOT, "configs", "vcara_full.yaml")) as f:
        y = yaml.safe_load(f)
    cfg = make_cfg("vcara")
    for k in ("gamma", "tau", "batch", "buffer", "ensemble", "fixed_horizon", "cf_margin",
              "cf_uncertainty_limit", "cf_target_rate", "cf_warmup", "cf_ramp", "beta", "lam"):
        assert getattr(cfg, k) == y[k], k


def test_all_variants_build():
    for v in VARIANTS:
        make_cfg(v)
