"""Short environment and agent smoke tests (CPU, a few steps)."""
import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("Box2D")

from vcara import make_cfg, CarRacingPixels  # noqa: E402


def test_env_shapes_and_action_mapping():
    cfg = make_cfg("vcara")
    env = CarRacingPixels(cfg, early_stop=False)
    obs, f0 = env.reset(seed=0)
    assert obs.shape == (cfg.stack, cfg.img, cfg.img)
    a = env.to_env_action(np.array([0.0, 0.0, -1.0], dtype=np.float32))
    assert a[2] == 0.0          # brake below dead zone maps to zero
    obs, r, terminal, ended, info = env.step(np.array([0.0, 0.5, -1.0], dtype=np.float32))
    assert obs.shape == (cfg.stack, cfg.img, cfg.img)
    env.env.close()
