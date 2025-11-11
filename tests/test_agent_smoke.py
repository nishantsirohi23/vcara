"""One optimisation step on random data (CPU)."""
import numpy as np
import pytest

pytest.importorskip("Box2D")

import torch  # noqa: E402
from vcara import make_cfg, VCARAAgent  # noqa: E402


def test_one_update_runs():
    cfg = make_cfg("vcara", batch=8, buffer=200, cf_warmup=0, cf_ramp=1)
    agent = VCARAAgent(cfg, "cpu")
    B, K, S = 8, cfg.stack, cfg.img
    nH = len(agent.HS)
    b = dict(
        obs=torch.randint(0, 255, (B, K, S, S), dtype=torch.uint8),
        hobs=torch.randint(0, 255, (B, nH, K, S, S), dtype=torch.uint8),
        act=torch.rand(B, 3) * 2 - 1,
        rew=torch.rand(B, agent.seq_len),
        fail=torch.zeros(B, agent.seq_len),
        rem=torch.full((B,), 50, dtype=torch.long),
        term=torch.zeros(B),
        fmc=torch.rand(B),
        cov=torch.rand(B),
        fin=torch.zeros(B),
        src=torch.zeros(B, dtype=torch.long),
        tf=torch.rand(B),
        htf=torch.rand(B, nH),
    )
    agent.update(b)
    m = agent.pop_metrics()
    assert np.isfinite(m["loss_q"])
