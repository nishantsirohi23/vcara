"""V-CARA V3.1 core: config, environment wrapper, replay, networks, agent, evaluation, training.

Ported from the V-CARA V3.1 training script. The PPO baseline is not included here yet.
Planned split: env.py, replay.py, networks.py, agent.py, train.py.
"""
import os, json, time, math, copy, random
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import gymnasium as gym

ENV_ID = "CarRacing-v3"


@dataclass
class Cfg:
    # --- environment ---
    action_repeat: int = 4
    img: int = 64
    stack: int = 4
    reward_scale: float = 0.1
    stall_frames: int = 160
    completion_bonus: float = 50.0
    time_limit: int = 1000
    time_aware: bool = True
    stall_mode: str = "terminal_penalty"       # "terminal_penalty" | "truncate"
    stall_bleed_frames: int = 400
    brake_deadzone: float = 0.6
    # --- optimisation ---
    total_frames: int = 2_000_000
    start_steps: int = 5_000
    batch: int = 256
    buffer: int = 300_000
    updates_per_step: float = 1.0
    gamma: float = 0.99
    tau: float = 0.005
    enc_tau: float = 0.02
    actor_lr: float = 1e-4
    critic_lr: float = 2e-4
    init_alpha: float = 0.1
    target_entropy: float = -3.0
    aug_pad: int = 4
    grad_clip: float = 10.0
    # --- network ---
    encoder: str = "stack"                     # "stack" | "gru"
    feat: int = 128
    gru_hidden: int = 128
    mlp: int = 256
    # --- V-CARA switches ---
    use_cf: bool = True
    use_gate: bool = True
    use_risk: bool = True
    adaptive_h: bool = False
    use_progress: bool = True
    use_finish: bool = True
    use_elite_replay: bool = True
    # --- horizons ---
    horizons: tuple = (2, 4, 8)
    fixed_horizon: int = 3
    u_limit: float = 1.0
    # --- counterfactual ---
    ensemble: int = 3
    cand_deltas: tuple = ((0, 0, 0), (-.1, 0, 0), (.1, 0, 0), (-.2, 0, 0), (.2, 0, 0),
                          (-.4, 0, 0), (.4, 0, 0), (-.6, 0, 0), (.6, 0, 0),
                          (0, -.4, 0), (0, .4, 0))      # index 0 MUST be nominal
    beta: float = 0.3
    lam: float = 0.01
    cf_margin: float = 0.02
    cf_uncertainty_limit: float = 1.5
    cf_target_rate: float = 0.25
    cf_scale: float = 0.20
    cf_warmup: int = 10_000
    cf_ramp: int = 10_000
    risk_horizon: int = 16
    gate_eps: float = 1e-6
    # --- elite replay / aux losses ---
    near_finish_threshold: float = 0.75
    elite_threshold: float = 0.90
    elite_offroad_max: float = 0.15
    normal_ratio: float = 0.625
    near_finish_ratio: float = 0.25
    elite_ratio: float = 0.125
    lambda_elite: float = 0.30
    lambda_finish: float = 0.20
    late_threshold: float = 0.75
    finish_label_cov: float = 0.90
    finish_gate: float = 0.02
    finish_gate_bacc: float = 0.70
    finish_window: int = 150
    # --- logging / eval / ckpt ---
    eval_every: int = 25_000
    eval_episodes: int = 10
    eval_stoch_episodes: int = 5
    eval_seeds: tuple = tuple(range(1000, 1010))
    final_eval_seeds: tuple = tuple(range(2000, 2020))
    log_every: int = 2_000
    ckpt_every: int = 100_000
    ret_ema_beta: float = 0.6

    @property
    def needs_dyn(self):
        return self.use_cf or self.adaptive_h

    @property
    def stall_steps(self):
        return int(math.ceil(self.stall_frames / self.action_repeat))


_OFF = dict(use_cf=False, adaptive_h=False, use_gate=False, use_risk=False,
            use_progress=False, use_finish=False, use_elite_replay=False)
VARIANTS = {
    "sac": dict(_OFF, fixed_horizon=1),
    "sac_n3": dict(_OFF, fixed_horizon=3),
    "sac_n5": dict(_OFF, fixed_horizon=5),
    "vcara": dict(),
    "vcara_stall_trunc": dict(stall_mode="truncate"),
    "vcara_no_cf": dict(use_cf=False),
    "vcara_no_elite": dict(use_elite_replay=False),
    "vcara_no_progress": dict(use_progress=False),
    "vcara_no_finish": dict(use_finish=False),
    "vcara_no_risk": dict(use_risk=False),
    "vcara_no_time": dict(time_aware=False),
    "vcara_gru": dict(encoder="gru"),
    "vcara_adaptive": dict(adaptive_h=True, fixed_horizon=4),
    "vcara_aggressive": dict(cf_margin=0.0, cf_uncertainty_limit=1e9, cf_target_rate=1.0),
}


def make_cfg(variant="vcara", **overrides):
    kw = dict(VARIANTS[variant])
    kw.update(overrides)
    return Cfg(**kw)


# ============================================================================
# Environment
# ============================================================================
def _offroad(u):
    return all(len(w.tiles) == 0 for w in u.car.wheels)


def _coverage(u):
    return u.tile_visited_count / max(len(u.track), 1)


class CarRacingPixels:
    """Gray 64x64 frames, K-frame stack, action repeat, completion bonus, stall cutoff.

    step() returns (obs, scaled_reward, terminal, ended, info):
      terminal = true MDP terminal (lap, out-of-field, and the time limit when
                 cfg.time_aware) -> no bootstrapping past it
      ended    = episode over for any reason (terminal, time limit, stall cut)
    """
    BRAKE_SCALE = 0.5

    def __init__(self, cfg: Cfg, early_stop=True, render_mode=None):
        self.cfg = cfg
        self.env = gym.make(ENV_ID, continuous=True, render_mode=render_mode)
        self.early_stop = early_stop
        self.stack = deque(maxlen=cfg.stack)
        self.video = None

    @property
    def tfrac(self):
        return min(self.n / self.cfg.time_limit, 1.0)

    def _proc(self, o):
        g = cv2.cvtColor(o, cv2.COLOR_RGB2GRAY)[:84]
        return cv2.resize(g, (self.cfg.img, self.cfg.img), interpolation=cv2.INTER_AREA)

    def reset(self, seed=None):
        o, _ = self.env.reset(seed=seed)
        f = self._proc(o)
        self.stack.clear()
        for _ in range(self.cfg.stack):
            self.stack.append(f)
        self.ret = 0.0; self.off = 0; self.n = 0; self.since_pos = 0; self.oof = False
        self.spd = 0.0; self.nsteps = 0
        self.spd_hist, self.cov_hist, self.fr_hist = [], [], []
        return np.stack(self.stack), f

    def to_env_action(self, a):
        dz = self.cfg.brake_deadzone
        gas = (a[1] + 1) / 2
        brake = max(0.0, (a[2] - dz) / (1.0 - dz)) * self.BRAKE_SCALE
        return np.array([a[0], gas, brake], dtype=np.float32)

    def step(self, a):
        act = self.to_env_action(np.clip(a, -1, 1))
        u = self.env.unwrapped
        tot, frames, fail, oof, lap, tlim = 0.0, 0, False, False, False, False
        for _ in range(self.cfg.action_repeat):
            o, r, te, tr, inf = self.env.step(act)
            if self.video is not None:
                self.video.append(self.env.render())
            frames += 1; tot += r; self.n += 1
            off = _offroad(u); self.off += int(off); fail |= off
            if te:
                if inf.get("lap_finished") is False:
                    oof = True; fail = True
                else:
                    lap = True
            if tr:
                tlim = True
            if te or tr:
                break
        if oof:
            self.oof = True
        if lap:
            tot += self.cfg.completion_bonus
        self.ret += tot
        v = u.car.hull.linearVelocity
        self.spd += math.hypot(v[0], v[1]); self.nsteps += 1
        self.since_pos = 0 if tot > 0.2 else self.since_pos + 1
        cov = _coverage(u)
        self.spd_hist.append(math.hypot(v[0], v[1])); self.cov_hist.append(cov); self.fr_hist.append(self.n)
        ended = oof or lap or tlim
        stalled = False
        if self.early_stop and not ended and self.since_pos > self.cfg.stall_steps:
            ended = True; stalled = True
        terminal = oof or lap or (tlim and self.cfg.time_aware)
        r_train = tot * self.cfg.reward_scale
        if stalled and self.cfg.stall_mode == "terminal_penalty":
            # A stall cut is training-only; as a terminal it must not look cheaper than
            # driving, so charge the discounted per-frame bleed the agent would escape.
            terminal = True
            left = max(0, min(self.cfg.time_limit - self.n, self.cfg.stall_bleed_frames))
            nst = int(math.ceil(left / self.cfg.action_repeat))
            g = self.cfg.gamma
            bleed = 0.1 * self.cfg.action_repeat * self.cfg.reward_scale
            r_train -= bleed * (1.0 - g ** nst) / (1.0 - g)
        f = self._proc(o)
        self.stack.append(f)
        info = dict(frame=f, frames=frames, fail=bool(fail), stalled=stalled, timeout=tlim, coverage=cov)
        if ended:
            info["ep_stats"] = dict(ret=self.ret, off_road_frac=self.off / max(self.n, 1),
                                    out_of_field=self.oof, lap_finished=lap,
                                    coverage=cov, frames=self.n,
                                    mean_speed=self.spd / max(self.nsteps, 1),
                                    stalled=stalled, **self._speed_stats())
        return np.stack(self.stack), r_train, terminal, ended, info

    def _speed_stats(self):
        sp = np.asarray(self.spd_hist, dtype=np.float64); cv = np.asarray(self.cov_hist)
        fr = np.asarray(self.fr_hist)
        if len(sp) == 0:
            return {}
        late = cv >= self.cfg.late_threshold
        h = max(len(sp) // 2, 1)
        t_to = lambda x: float(fr[np.argmax(cv >= x)]) if (cv >= x).any() else -1.0
        return dict(speed_p50=float(np.percentile(sp, 50)), speed_p90=float(np.percentile(sp, 90)),
                    speed_max=float(sp.max()),
                    speed_late=float(sp[late].mean()) if late.any() else -1.0,
                    speed_h1=float(sp[:h].mean()),
                    speed_h2=float(sp[h:].mean()) if len(sp) > h else float(sp.mean()),
                    t50=t_to(0.5), t75=t_to(0.75), t90=t_to(0.9))


def random_action(rng):
    return np.array([rng.uniform(-1, 1), rng.uniform(-0.3, 1), rng.uniform(-1, -0.6)], dtype=np.float32)


# ============================================================================
# Replay: device-resident ring buffers, vectorised sampling
# ============================================================================
class FrameRing:
    """Episodes stored back-to-back in preallocated tensors. An episode with T steps
    occupies T+1 frame slots; slot s+t holds frame t, action/reward/fail of step t."""
    def __init__(self, capacity, S, dev, slack=2048):
        self.cap = capacity + slack
        self.dev = dev
        self.frames = torch.zeros((self.cap, S, S), dtype=torch.uint8, device=dev)
        self.act = torch.zeros((self.cap, 3), dtype=torch.float32, device=dev)
        self.rew = torch.zeros(self.cap, dtype=torch.float32, device=dev)
        self.fail = torch.zeros(self.cap, dtype=torch.float32, device=dev)
        self.cov = torch.zeros(self.cap, dtype=torch.float32, device=dev)
        self.fint = torch.zeros(self.cap, dtype=torch.float32, device=dev)
        self.ptr = 0
        self.eps = deque()      # (start, T, done, max_cov)
        self.size = 0
        self._refresh()

    def _refresh(self):
        if self.eps:
            a = np.array(self.eps, dtype=np.float64)
            self.starts = a[:, 0].astype(np.int64); self.Ts = a[:, 1].astype(np.int64)
            self.done = a[:, 2].astype(np.float32); self.fmc = a[:, 3].astype(np.float32)
            self.cum = np.cumsum(self.Ts); self.size = int(self.cum[-1])
        else:
            self.size = 0

    def add(self, ft, at, rt, flt, done, fmc, cov, fint):
        T = int(at.shape[0]); n = T + 1
        if self.ptr + n > self.cap:
            while self.eps and self.eps[0][0] >= self.ptr:
                self.eps.popleft()
            self.ptr = 0
        s, e = self.ptr, self.ptr + n
        while self.eps and self.eps[0][0] < e and self.eps[0][0] + self.eps[0][1] + 1 > s:
            self.eps.popleft()
        self.frames[s:e] = ft
        self.act[s:s + T] = at; self.rew[s:s + T] = rt; self.fail[s:s + T] = flt
        self.cov[s:s + T] = cov; self.fint[s:s + T] = fint
        self.eps.append((s, T, float(done), float(fmc)))
        self.ptr = e
        self._refresh()

    def sample(self, n, K, HS, seq_len, ar, tlim, rng):
        dev = self.dev
        u = rng.integers(0, self.size, size=n)
        ei = np.searchsorted(self.cum, u, side="right")
        T = self.Ts[ei]; st = self.starts[ei]
        t = u - (self.cum[ei] - T)
        base = np.arange(K) - (K - 1)
        oi = st[:, None] + np.maximum(t[:, None] + base, 0)
        te = np.minimum(t[:, None] + HS[None, :], T[:, None])
        hi = st[:, None, None] + np.maximum(te[..., None] + base, 0)
        L = np.arange(seq_len)
        rmask = L[None, :] < (T - t)[:, None]
        ri = np.where(rmask, st[:, None] + t[:, None] + L[None, :], (st + t)[:, None])
        tt = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(dev)
        rm = tt(rmask.astype(np.float32))
        return dict(
            obs=self.frames[tt(oi)], hobs=self.frames[tt(hi)], act=self.act[tt(st + t)],
            rew=self.rew[tt(ri)] * rm, fail=self.fail[tt(ri)] * rm,
            rem=tt(T - t), term=tt(self.done[ei]), fmc=tt(self.fmc[ei]),
            cov=self.cov[tt(st + t)], fin=self.fint[tt(st + t)],
            tf=tt(np.minimum(t * ar / tlim, 1.0).astype(np.float32)),
            htf=tt(np.minimum(te * ar / tlim, 1.0).astype(np.float32)))


class EpisodeReplay:
    """Normal FIFO ring plus retention rings for near-finish and elite episodes.
    Classification uses coverage and off-road fraction only."""
    def __init__(self, capacity, cfg: Cfg, S, dev):
        self.cfg = cfg
        self.rings = {"normal": FrameRing(capacity, S, dev),
                      "near_finish": FrameRing(capacity // 8, S, dev),
                      "elite": FrameRing(capacity // 8, S, dev)}
        self.n_eps = defaultdict(int)

    def add(self, frames, actions, rewards, fail, done, max_cov, off_frac, lap, cov_traj):
        T = len(actions)
        if T == 0:
            return
        c = self.cfg
        dev = self.rings["normal"].dev
        ft = torch.from_numpy(np.asarray(frames, dtype=np.uint8)).to(dev)
        at = torch.from_numpy(np.asarray(actions, dtype=np.float32)).to(dev)
        rt = torch.from_numpy(np.asarray(rewards, dtype=np.float32)).to(dev)
        flt = torch.from_numpy(np.asarray(fail, dtype=np.float32)).to(dev)
        cv = np.asarray(cov_traj, dtype=np.float32)
        cov_t = cv[:T]
        fint = (cv[np.minimum(np.arange(T) + c.finish_window, T)] >= c.finish_label_cov).astype(np.float32)
        dst = ["normal"]
        if max_cov >= c.near_finish_threshold:
            dst.append("near_finish")
        if max_cov >= c.elite_threshold and off_frac <= c.elite_offroad_max:
            dst.append("elite")
        for b in dst:
            self.rings[b].add(ft, at, rt, flt, done, max_cov,
                              torch.from_numpy(cov_t).to(dev), torch.from_numpy(fint).to(dev))
            self.n_eps[b] += 1

    def sample(self, B, K, HS, seq_len, rng):
        c = self.cfg
        R = self.rings
        near_ok = c.use_elite_replay and R["near_finish"].size > 500
        elite_ok = c.use_elite_replay and R["elite"].size > 200
        want_elite = min(int(round(c.elite_ratio * B)), B) if elite_ok else 0
        want_near = min(int(round(c.near_finish_ratio * B)), B - want_elite) if near_ok else 0
        want_norm = B - want_near - want_elite
        parts, srcs = [], []
        for b, n, sid in (("normal", want_norm, 0), ("near_finish", want_near, 1), ("elite", want_elite, 2)):
            if n > 0:
                parts.append(R[b].sample(n, K, HS, seq_len, c.action_repeat, c.time_limit, rng))
                srcs.append(torch.full((n,), sid, dtype=torch.int64, device=R[b].dev))
        out = {k: torch.cat([p[k] for p in parts], 0) for k in parts[0]}
        out["src"] = torch.cat(srcs, 0)
        return out

    @property
    def size(self):
        return self.rings["normal"].size


# ============================================================================
# Networks
# ============================================================================
def mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n):
        layers += [nn.Linear(d, h), nn.ReLU()]; d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


def random_shift(x, pad=4):
    n, c, h, w = x.shape
    x = F.pad(x, (pad,) * 4, mode="replicate")
    eps = 1.0 / (h + 2 * pad)
    ar = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad, device=x.device, dtype=x.dtype)[:h]
    ar = ar.unsqueeze(0).repeat(h, 1).unsqueeze(2)
    base = torch.cat([ar, ar.transpose(1, 0)], dim=2).unsqueeze(0).repeat(n, 1, 1, 1)
    shift = torch.randint(0, 2 * pad + 1, (n, 1, 1, 2), device=x.device, dtype=x.dtype) * (2.0 / (h + 2 * pad))
    return F.grid_sample(x, base + shift, padding_mode="zeros", align_corners=False)


class Encoder(nn.Module):
    """'stack': K frames as K input channels -> z (fast). 'gru': per-frame CNN + GRU.
    Output = [z, (gru h)] + optional remaining-time fraction. z is always first (feat dims)."""
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.kind, self.time_aware, self.feat = cfg.encoder, cfg.time_aware, cfg.feat
        cin = cfg.stack if self.kind == "stack" else 1
        self.cnn = nn.Sequential(
            nn.Conv2d(cin, 32, 3, 2), nn.ReLU(), nn.Conv2d(32, 32, 3, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 1), nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            n = self.cnn(torch.zeros(1, cin, cfg.img, cfg.img)).shape[1]
        self.fc = nn.Sequential(nn.Linear(n, cfg.feat), nn.LayerNorm(cfg.feat), nn.Tanh())
        if self.kind == "gru":
            self.gru = nn.GRU(cfg.feat, cfg.gru_hidden, batch_first=True)
            self.out_dim = cfg.feat + cfg.gru_hidden
        else:
            self.out_dim = cfg.feat
        self.out_dim += int(self.time_aware)

    def forward(self, x, tf):
        B, K, H, W = x.shape
        x = x.float() / 255.0 - 0.5
        if self.kind == "stack":
            s = self.fc(self.cnn(x))
        else:
            z = self.fc(self.cnn(x.reshape(B * K, 1, H, W))).view(B, K, -1)
            out, _ = self.gru(z)
            s = torch.cat([z[:, -1], out[:, -1]], dim=-1)
        if self.time_aware:
            s = torch.cat([s, tf.unsqueeze(-1)], dim=-1)
        return s


class TwinQ(nn.Module):
    def __init__(self, ds, h, adim=3):
        super().__init__(); self.q1 = mlp(ds + adim, h, 1); self.q2 = mlp(ds + adim, h, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], -1); return self.q1(x), self.q2(x)


class Actor(nn.Module):
    LOG_MIN, LOG_MAX = -10.0, 2.0

    def __init__(self, ds, h, adim=3):
        super().__init__(); self.net = mlp(ds, h, 2 * adim)

    def forward(self, s, deterministic=False):
        mu, ls = self.net(s).chunk(2, -1)
        ls = self.LOG_MIN + 0.5 * (self.LOG_MAX - self.LOG_MIN) * (torch.tanh(ls) + 1)
        if deterministic:
            return torch.tanh(mu), None
        eps = torch.randn_like(mu); u = mu + ls.exp() * eps; a = torch.tanh(u)
        logp = (-0.5 * eps.pow(2) - ls - 0.5 * math.log(2 * math.pi)).sum(-1)
        logp = logp - (2 * (math.log(2) - u - F.softplus(-2 * u))).sum(-1)
        return a, logp


class EnsLinear(nn.Module):
    def __init__(self, M, i, o):
        super().__init__()
        self.w = nn.Parameter(torch.randn(M, i, o) / math.sqrt(i)); self.b = nn.Parameter(torch.zeros(M, 1, o))

    def forward(self, x):
        return torch.baddbmm(self.b, x, self.w)


class DynamicsEnsemble(nn.Module):
    """M MLPs: (s_t, a) -> z_{t+h} for each h in the horizon grid (residual on z_t)."""
    def __init__(self, ds, feat, nH, M, h):
        super().__init__()
        self.M, self.nH, self.feat = M, nH, feat
        self.net = nn.Sequential(EnsLinear(M, ds + 3, h), nn.ReLU(), EnsLinear(M, h, h), nn.ReLU(),
                                 EnsLinear(M, h, nH * feat))

    def forward(self, s, a):
        x = torch.cat([s, a], -1).unsqueeze(0).expand(self.M, -1, -1)
        out = self.net(x).view(self.M, s.shape[0], self.nH, self.feat)
        return out + s[:, :self.feat].unsqueeze(0).unsqueeze(2)


def soft_update(tgt, src, tau):
    with torch.no_grad():
        for pt, p in zip(tgt.parameters(), src.parameters()):
            pt.lerp_(p, tau)


def flatten_gru(mod):
    for m in mod.modules():
        if isinstance(m, nn.GRU):
            m.flatten_parameters()


# ============================================================================
# Agent
# ============================================================================
class VCARAAgent:
    def __init__(self, cfg: Cfg, device):
        self.cfg, self.dev = cfg, torch.device(device)
        c = cfg
        self.enc = Encoder(c).to(self.dev); self.enc_t = copy.deepcopy(self.enc)
        flatten_gru(self.enc); flatten_gru(self.enc_t)
        ds = self.enc.out_dim
        self.q = TwinQ(ds, c.mlp).to(self.dev); self.q_t = copy.deepcopy(self.q)
        self.actor = Actor(ds, c.mlp).to(self.dev)
        self.log_alpha = torch.tensor(math.log(c.init_alpha), device=self.dev, requires_grad=True)

        self.HS = tuple(c.horizons) if c.adaptive_h else (c.fixed_horizon,)
        if c.adaptive_h:
            assert c.fixed_horizon in self.HS, "fixed_horizon must be in horizons when adaptive_h"
        self.fixed_idx = self.HS.index(c.fixed_horizon) if c.fixed_horizon in self.HS else 0
        self.HS_np = np.array(self.HS, dtype=np.int64)
        self.HS_t = torch.tensor(self.HS, device=self.dev, dtype=torch.long)
        self.seq_len = max(max(self.HS), c.risk_horizon)

        self.dyn = DynamicsEnsemble(ds, c.feat, len(self.HS), c.ensemble, c.mlp).to(self.dev) if c.needs_dyn else None
        self.risk = mlp(ds + 3, c.mlp, 1).to(self.dev) if (c.use_cf and c.use_risk) else None
        self.progress = mlp(ds, c.mlp, 1).to(self.dev) if c.use_progress else None
        self.finish = mlp(ds + 3, c.mlp, 1).to(self.dev) if c.use_finish else None
        self.deltas = torch.tensor(c.cand_deltas, dtype=torch.float32, device=self.dev)
        self.disc = c.gamma ** torch.arange(self.seq_len, device=self.dev, dtype=torch.float32)

        self.critic_params = (list(self.enc.parameters()) + list(self.q.parameters())
                              + (list(self.progress.parameters()) if self.progress is not None else []))
        self.critic_opt = torch.optim.Adam(self.critic_params, lr=c.critic_lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=c.actor_lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=c.actor_lr)
        self.dyn_opt = torch.optim.Adam(self.dyn.parameters(), lr=c.critic_lr) if self.dyn is not None else None
        self.risk_opt = torch.optim.Adam(self.risk.parameters(), lr=c.critic_lr) if self.risk is not None else None
        self.fin_opt = torch.optim.Adam(self.finish.parameters(), lr=c.critic_lr) if self.finish is not None else None

        self.c_ref = None; self.u_ref = None                    # device scalars (no host syncs)
        self.fin_pos_ema = torch.zeros((), device=self.dev)
        self.fin_bacc = torch.zeros((), device=self.dev)
        self.updates = 0
        self._acc = defaultdict(float); self._n = 0

    def opts(self):
        d = dict(critic=self.critic_opt, actor=self.actor_opt, alpha=self.alpha_opt)
        for k, o in (("dyn", self.dyn_opt), ("risk", self.risk_opt), ("fin", self.fin_opt)):
            if o is not None:
                d[k] = o
        return d

    # ---------------------------------------------------------------- acting
    @torch.no_grad()
    def act(self, obs, tf, deterministic=False):
        x = torch.from_numpy(obs).unsqueeze(0).to(self.dev)
        t = torch.tensor([tf], dtype=torch.float32, device=self.dev)
        a, _ = self.actor(self.enc(x, t), deterministic)
        return a[0].cpu().numpy()

    # ----------------------------------------------------- counterfactual block
    @torch.no_grad()
    def _counterfactual(self, sd, a_nom):
        c = self.cfg; B = sd.shape[0]; Kc = self.deltas.shape[0]; nH = len(self.HS); M = c.ensemble
        cand = (a_nom.unsqueeze(1) + self.deltas.unsqueeze(0)).clamp(-1, 1)
        sr = sd.unsqueeze(1).expand(-1, Kc, -1).reshape(B * Kc, -1)
        cr = cand.reshape(B * Kc, 3)

        q1, q2 = self.q(sr, cr); qv = torch.min(q1, q2).view(B, Kc)
        risk = torch.sigmoid(self.risk(torch.cat([sr, cr], -1))).view(B, Kc) if self.risk is not None \
            else torch.zeros_like(qv)

        pred = self.dyn(sr, cr).view(M, B, Kc, nH, c.feat)
        mu, var = pred.mean(0), pred.var(0, unbiased=False)
        U = var.mean((-1, -2))                                  # (B,Kc)
        C2 = ((mu - mu[:, :1]) ** 2).mean(-1)                   # (B,Kc,nH)

        c_now = C2[:, 1:, :].mean(); u_now = U.mean()
        self.c_ref = c_now if self.c_ref is None else 0.99 * self.c_ref + 0.01 * c_now
        self.u_ref = u_now if self.u_ref is None else 0.99 * self.u_ref + 0.01 * u_now

        score = qv - c.beta * risk - c.lam * U / (self.u_ref + 1e-8)
        j = score.argmax(1)
        ar = torch.arange(B, device=sd.device)
        a_star = cand[ar, j]
        score_nom, score_star = score[:, 0], score[ar, j]

        # Conservative acceptance: clearly better, at most the top `cf_target_rate`
        # of the batch, and prediction spread not above cf_uncertainty_limit * average.
        improvement = score_star - score_nom
        u_norm_star = U[ar, j] / (self.u_ref + 1e-8)
        thr = torch.clamp(torch.quantile(improvement, 1.0 - c.cf_target_rate), min=c.cf_margin)
        cf_ok = (improvement > thr) & (u_norm_star < c.cf_uncertainty_limit)
        a_target = torch.where(cf_ok.unsqueeze(-1), a_star, a_nom)

        if c.adaptive_h and nH > 1:
            U_hn = var.mean((1, 3)) / (self.u_ref + 1e-8)       # (B,nH)
            ok_rev = (U_hn < c.u_limit).flip(1)
            any_ok = ok_rev.any(1)
            first_ok = ok_rev.float().argmax(1)
            idx = torch.where(any_ok, (nH - 1) - first_ok, torch.zeros_like(first_ok))
        else:
            idx = torch.zeros(B, dtype=torch.long, device=sd.device)

        c2s, us = C2[ar, j, -1], U[ar, j]
        cn, un = c2s / (self.c_ref + 1e-8), us / (self.u_ref + 1e-8)
        G = cn / (cn + un + c.gate_eps)
        return dict(a_target=a_target, G=G, sel=idx, changed=cf_ok.float(), U=us,
                    C=C2[ar, j, -1].sqrt(), improvement=improvement, u_norm=u_norm_star)

    # ---------------------------------------------------------------- update
    def update(self, b):
        c, dev = self.cfg, self.dev
        obs, hobs, act, rew, fail = b["obs"], b["hobs"], b["act"], b["rew"], b["fail"]
        rem, term, src, fmc, fin, tf, htf = b["rem"], b["term"], b["src"], b["fmc"], b["fin"], b["tf"], b["htf"]
        late = (b["cov"] >= c.late_threshold).float()
        B, nH, K = obs.shape[0], len(self.HS), c.stack
        S = obs.shape[-1]

        s = self.enc(random_shift(obs.float(), c.aug_pad), tf)
        sd = s.detach()
        a_pi, logp = self.actor(sd)
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            s_h = self.enc_t(hobs.reshape(B * nH, K, S, S), htf.reshape(B * nH))
            a_h, logp_h = self.actor(s_h)
            q1t, q2t = self.q_t(s_h, a_h)
            v_h = (torch.min(q1t, q2t).squeeze(-1) - alpha * logp_h).view(B, nH)
            z_h = s_h[:, :c.feat].view(B, nH, c.feat)
            cumR = torch.cumsum(rew * self.disc, 1)[:, self.HS_t - 1]
            steps = torch.minimum(self.HS_t.unsqueeze(0), rem.unsqueeze(1)).float()
            done_h = (term.unsqueeze(1) > 0.5) & (rem.unsqueeze(1) <= self.HS_t.unsqueeze(0))
            y_all = cumR + (c.gamma ** steps) * (~done_h).float() * v_h

        cf = None
        if self.dyn is not None and self.updates >= c.cf_warmup:
            cf = self._counterfactual(sd, a_pi.detach())
        sel = cf["sel"] if (c.adaptive_h and cf is not None) \
            else torch.full((B,), self.fixed_idx, device=dev, dtype=torch.long)
        y = y_all.gather(1, sel.unsqueeze(1)).squeeze(1)

        # ---- critic (+ progress aux) ----
        q1, q2 = self.q(s, act)
        loss_q = F.mse_loss(q1.squeeze(-1), y) + F.mse_loss(q2.squeeze(-1), y)
        loss_c = loss_q
        if self.progress is not None:
            pr = torch.sigmoid(self.progress(s).squeeze(-1))
            loss_c = loss_c + F.mse_loss(pr, fmc)
        self.critic_opt.zero_grad(set_to_none=True); loss_c.backward()
        nn.utils.clip_grad_norm_(self.critic_params, c.grad_clip); self.critic_opt.step()
        flatten_gru(self.enc)

        # ---- actor ----
        q1p, q2p = self.q(sd, a_pi)
        l_pol = alpha * logp - torch.min(q1p, q2p).squeeze(-1)
        aux = {}
        if c.use_cf and cf is not None:
            l_cf = ((a_pi - cf["a_target"].detach()) ** 2).sum(-1)
            ramp = min(max((self.updates - c.cf_warmup) / max(c.cf_ramp, 1), 0.0), 1.0)
            Gw = (cf["G"] if c.use_gate else torch.ones_like(cf["G"])) * ramp
            l_pol = l_pol + c.cf_scale * Gw * l_cf
            aux["cf_ramp"] = torch.tensor(ramp, device=dev)
            acc = cf["changed"]
            aux["cf_acc_late"] = (acc * late).sum() / late.sum().clamp(min=1)
            aux["cf_acc_early"] = (acc * (1 - late)).sum() / (1 - late).sum().clamp(min=1)
        if self.finish is not None:
            logit_a = self.finish(torch.cat([sd, a_pi], -1)).squeeze(-1)
            l_fin = -(F.logsigmoid(logit_a) * late).sum() / late.sum().clamp(min=1)
            gate = ((self.fin_pos_ema > c.finish_gate) & (self.fin_bacc > c.finish_gate_bacc)
                    & (late.sum() > 0)).float()
            l_pol = l_pol + c.lambda_finish * gate * l_fin
            aux["finish_p"] = torch.sigmoid(logit_a).detach().mean(); aux["late_frac"] = late.mean()
            aux["fin_gate"] = gate
        if c.use_elite_replay:
            elite = (src == 2).float()
            l_elite = ((a_pi - act) ** 2).sum(-1)
            l_pol = l_pol + c.lambda_elite * elite * l_elite
            aux["elite_frac"] = elite.mean(); aux["near_frac"] = (src == 1).float().mean()
        self.actor_opt.zero_grad(set_to_none=True); l_pol.mean().backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), c.grad_clip); self.actor_opt.step()

        loss_a = -(self.log_alpha * (logp.detach() + c.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True); loss_a.backward(); self.alpha_opt.step()

        m = {"loss_q": loss_q.detach(), "q": q1.detach().mean(), "y": y.mean(), "alpha": alpha,
             "l_pol": l_pol.detach().mean(), "H": self.HS_t[sel].float().mean()}
        m["q_gap"] = (q1.detach().squeeze(-1) - y).abs().mean()
        if self.progress is not None:
            m["prog_loss"] = F.mse_loss(torch.sigmoid(self.progress(sd).squeeze(-1)), fmc).detach()
        if self.dyn is not None:
            pred = self.dyn(sd, act)
            hv = (self.HS_t.unsqueeze(0) <= rem.unsqueeze(1)).float()
            mask = (torch.rand(c.ensemble, B, 1, device=dev) < 0.8).float() * hv.unsqueeze(0)
            se = ((pred - z_h.unsqueeze(0)) ** 2).mean(-1)
            loss_d = (se * mask).sum() / mask.sum().clamp(min=1.0)
            self.dyn_opt.zero_grad(set_to_none=True); loss_d.backward(); self.dyn_opt.step()
            with torch.no_grad():
                pers = ((sd[:, :c.feat].unsqueeze(1) - z_h) ** 2).mean(-1)
                pers = (pers * hv).sum() / hv.sum().clamp(min=1.0)
            m["dyn_mse"] = loss_d.detach(); m["dyn_persist_mse"] = pers
        if self.risk is not None:
            label = (fail[:, :c.risk_horizon].sum(1) > 0).float()
            logit = self.risk(torch.cat([sd, act], -1)).squeeze(-1)
            loss_r = F.binary_cross_entropy_with_logits(logit, label)
            self.risk_opt.zero_grad(set_to_none=True); loss_r.backward(); self.risk_opt.step()
            m["risk_loss"] = loss_r.detach(); m["risk_pos_rate"] = label.mean()
        if self.finish is not None:
            logit_f = self.finish(torch.cat([sd, act], -1)).squeeze(-1)
            loss_f = F.binary_cross_entropy_with_logits(logit_f, fin)
            with torch.no_grad():   # metrics on the batch BEFORE this step trains on it
                pred = (logit_f > 0).float(); neg = 1.0 - fin
                tp = (pred * fin).sum(); fp = (pred * neg).sum(); tn = ((1 - pred) * neg).sum()
                tpr = tp / fin.sum().clamp(min=1.0); tnr = tn / neg.sum().clamp(min=1.0)
                bacc = 0.5 * (tpr + tnr)
                both = ((fin.sum() > 0) & (neg.sum() > 0)).float()
                self.fin_bacc = both * (0.99 * self.fin_bacc + 0.01 * bacc) + (1 - both) * self.fin_bacc
                m["fin_prec"] = tp / (tp + fp).clamp(min=1.0); m["fin_rec"] = tpr
                m["fin_acc"] = (pred == fin).float().mean(); m["fin_bacc"] = self.fin_bacc
            self.fin_opt.zero_grad(set_to_none=True); loss_f.backward(); self.fin_opt.step()
            self.fin_pos_ema = 0.99 * self.fin_pos_ema + 0.01 * fin.mean()
            m["fin_loss"] = loss_f.detach(); m["fin_pos_rate"] = fin.mean(); m["fin_pos_ema"] = self.fin_pos_ema

        if cf is not None:
            m.update(changed=cf["changed"].mean(), G=cf["G"].mean(), U_star=cf["U"].mean(),
                     C_star=cf["C"].mean(), cf_impr=cf["improvement"].mean(), cf_u=cf["u_norm"].mean())
        m.update(aux)
        m["late_frac_batch"] = late.mean()

        soft_update(self.q_t, self.q, c.tau); soft_update(self.enc_t, self.enc, c.enc_tau)
        self.updates += 1
        for k, v in m.items():
            self._acc[k] = self._acc[k] + v
        self._n += 1

    def pop_metrics(self):
        out = {k: float(v) / max(self._n, 1) for k, v in self._acc.items()}
        self._acc = defaultdict(float); self._n = 0
        return out

    # ---------------------------------------------------------------- io
    def save(self, path, extra=None):
        f = lambda x: None if x is None else float(x)
        d = dict(cfg=asdict(self.cfg), enc=self.enc.state_dict(), actor=self.actor.state_dict(),
                 q=self.q.state_dict(), q_t=self.q_t.state_dict(), enc_t=self.enc_t.state_dict(),
                 dyn=self.dyn.state_dict() if self.dyn is not None else None,
                 risk=self.risk.state_dict() if self.risk is not None else None,
                 progress=self.progress.state_dict() if self.progress is not None else None,
                 finish=self.finish.state_dict() if self.finish is not None else None,
                 log_alpha=self.log_alpha.detach().cpu(), updates=self.updates,
                 c_ref=f(self.c_ref), u_ref=f(self.u_ref),
                 fin_pos_ema=float(self.fin_pos_ema), fin_bacc=float(self.fin_bacc),
                 opts={k: o.state_dict() for k, o in self.opts().items()})
        if extra:
            d.update(extra)
        torch.save(d, path)

    def load_full(self, path):
        d = torch.load(path, map_location=self.dev, weights_only=False)
        self.enc.load_state_dict(d["enc"]); self.enc_t.load_state_dict(d["enc_t"])
        self.actor.load_state_dict(d["actor"]); self.q.load_state_dict(d["q"]); self.q_t.load_state_dict(d["q_t"])
        for name in ("dyn", "risk", "progress", "finish"):
            mod = getattr(self, name)
            if mod is not None and d.get(name) is not None:
                mod.load_state_dict(d[name])
        with torch.no_grad():
            self.log_alpha.copy_(d["log_alpha"].to(self.dev))
        self.updates = int(d.get("updates", 0))
        mk = lambda x: None if x is None else torch.tensor(x, device=self.dev)
        self.c_ref, self.u_ref = mk(d.get("c_ref")), mk(d.get("u_ref"))
        self.fin_pos_ema = torch.tensor(d.get("fin_pos_ema", 0.0), device=self.dev)
        self.fin_bacc = torch.tensor(d.get("fin_bacc", 0.0), device=self.dev)
        for k, o in self.opts().items():
            if d.get("opts", {}).get(k) is not None:
                o.load_state_dict(d["opts"][k])
        flatten_gru(self.enc); flatten_gru(self.enc_t)
        return d

    def load_policy(self, path):
        """Load encoder and actor only (for evaluation and video recording)."""
        d = torch.load(path, map_location=self.dev, weights_only=False)
        self.enc.load_state_dict(d["enc"]); self.actor.load_state_dict(d["actor"])
        flatten_gru(self.enc)
        return d


# ============================================================================
# Evaluation
# ============================================================================
def _vmean(stats, k):
    """Mean over episodes where the metric is defined (-1 = undefined); -1 if none."""
    v = [s[k] for s in stats if s.get(k, -1.0) >= 0]
    return float(np.mean(v)) if v else -1.0


def summarize_eps(stats):
    r = np.array([s["ret"] for s in stats])
    cov = np.array([s["coverage"] for s in stats])
    return dict(mean_return=float(r.mean()), std_return=float(r.std()),
                completion=float(np.mean([s["lap_finished"] for s in stats])),
                out_of_field=float(np.mean([s["out_of_field"] for s in stats])),
                off_road_frac=float(np.mean([s["off_road_frac"] for s in stats])),
                mean_speed=float(np.mean([s.get("mean_speed", 0.0) for s in stats])),
                mean_coverage=float(cov.mean()), max_coverage=float(cov.max()),
                median_coverage=float(np.median(cov)),
                p90_coverage=float(np.percentile(cov, 90)), n=len(stats),
                **{k: _vmean(stats, k) for k in ("speed_p50", "speed_p90", "speed_max", "speed_late",
                                                  "speed_h1", "speed_h2", "t50", "t75", "t90")})


def evaluate(agent, cfg, seeds, deterministic=True, record=False, ret_ema=None):
    env = CarRacingPixels(cfg, early_stop=False, render_mode="rgb_array" if record else None)
    if record:
        env.video = []
    stats = []
    for sd in seeds:
        obs, _ = env.reset(seed=int(sd))
        while True:
            obs, _, _, ended, info = env.step(agent.act(obs, env.tfrac, deterministic))
            if ended:
                stats.append(info["ep_stats"]); break
    out = summarize_eps(stats)
    if ret_ema is not None:
        prev = ret_ema.get("v")
        base = prev if prev is not None else out["mean_return"]
        out["ema_return"] = cfg.ret_ema_beta * base + (1 - cfg.ret_ema_beta) * out["mean_return"]
        ret_ema["v"] = out["ema_return"]
    frames = env.video
    env.env.close()
    return (out, frames) if record else out


def health_report(agent, replay, cfg, log):
    """One line per mechanism: is it alive?"""
    d = log["diag"][-5:]
    g = lambda k: float(np.mean([x[k] for x in d if k in x])) if any(k in x for x in d) else float("nan")
    parts = []
    if cfg.use_cf:
        ch = g("changed")
        inactive = agent.updates > cfg.cf_warmup + cfg.cf_ramp // 2 and not ch > 0.005
        parts.append(f"CF_accept={ch:.3f} (late={g('cf_acc_late'):.3f} early={g('cf_acc_early'):.3f}) "
                     f"impr={g('cf_impr'):.3f} u_norm={g('cf_u'):.2f}" + (" <-- INACTIVE" if inactive else ""))
    if cfg.use_elite_replay:
        parts.append(f"elite={replay.rings['elite'].size} near={replay.rings['near_finish'].size}")
    if cfg.use_finish:
        parts.append(f"fin_pos_ema={float(agent.fin_pos_ema):.3f} bacc={float(agent.fin_bacc):.2f} "
                     f"prec={g('fin_prec'):.2f} rec={g('fin_rec'):.2f} gate={g('fin_gate'):.2f} "
                     f"late={g('late_frac_batch'):.2f}")
    parts.append(f"q_gap={g('q_gap'):.3f} alpha={g('alpha'):.4f}")
    return "HEALTH " + " | ".join(parts)


# ============================================================================
# Training loop (crash-safe, disk-backed)
# ============================================================================
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _read_meta(ck_dir, tag):
    p = os.path.join(ck_dir, f"{tag}.json")
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:
            return None
    return None


def train(cfg: Cfg, name: str, seed: int, out_dir: str, device=None, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(dev).startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    env = CarRacingPixels(cfg, early_stop=True)
    agent = VCARAAgent(cfg, dev)
    replay = EpisodeReplay(cfg.buffer, cfg, cfg.img, torch.device(dev))
    rng = np.random.default_rng(seed)
    ck_dir = os.path.join(out_dir, "ckpts", f"{name}_seed{seed}")
    os.makedirs(ck_dir, exist_ok=True)
    log = dict(name=name, seed=seed, cfg=asdict(cfg), evals=[], episodes=[], diag=[], ckpts=[])
    path = os.path.join(out_dir, f"{name}_seed{seed}.json")

    frames_total = agent_steps = 0
    best_cov = saved_cov = 0.0
    t_last_best = 0.0
    ret_ema = {"v": None}
    meta = _read_meta(ck_dir, "periodic")
    if meta is not None and os.path.exists(os.path.join(ck_dir, "periodic.pt")):
        d = agent.load_full(os.path.join(ck_dir, "periodic.pt"))
        frames_total = int(meta.get("frames_total", d.get("frames_total", 0)))
        agent_steps = int(meta.get("agent_steps", d.get("agent_steps", 0)))
        best_cov = float(meta.get("best_cov", 0.0))
        saved_cov = best_cov; t_last_best = time.time()
        if os.path.exists(path):
            old = json.load(open(path))
            log["evals"] = [e for e in old.get("evals", []) if e.get("frames", 0) <= frames_total]
            log["episodes"] = [e for e in old.get("episodes", []) if e.get("at_frames", 0) <= frames_total]
            log["diag"] = [x for x in old.get("diag", []) if x.get("frames", 0) <= frames_total]
            log["ckpts"] = old.get("ckpts", [])
            for e in reversed(log["evals"]):
                if "ema_return" in e:
                    ret_ema["v"] = e["ema_return"]; break
        if verbose:
            print(f"[{name} s{seed}] RESUMED at {frames_total} frames (replay restarts empty)")
    next_eval = ((frames_total // cfg.eval_every) + 1) * cfg.eval_every
    next_ckpt = ((frames_total // cfg.ckpt_every) + 1) * cfg.ckpt_every
    obs, f0 = env.reset(seed=seed)
    ef, ea, er, efail = [f0], [], [], []
    ecov = [_coverage(env.env.unwrapped)]
    best_eval = dict(comp=-1.0, cov=-1.0)
    t0 = time.time()
    credit = 0.0

    def save_ckpt(tag):
        p = os.path.join(ck_dir, f"{tag}.pt")
        agent.save(p, extra=dict(frames_total=frames_total, agent_steps=agent_steps, tag=tag))
        json.dump(dict(frames_total=frames_total, agent_steps=agent_steps,
                       best_cov=best_cov, tag=tag), open(os.path.join(ck_dir, f"{tag}.json"), "w"))
        log["ckpts"].append(dict(tag=tag, frames=frames_total))
        json.dump(log, open(path, "w"))

    def do_eval(final=False):
        seeds = list(cfg.final_eval_seeds) if final else list(cfg.eval_seeds)[:cfg.eval_episodes]
        res = evaluate(agent, cfg, seeds, deterministic=True, ret_ema=ret_ema)
        if cfg.eval_stoch_episodes > 0:
            st = evaluate(agent, cfg, seeds[:cfg.eval_stoch_episodes], deterministic=False)
            res.update(stoch_return=st["mean_return"], stoch_coverage=st["mean_coverage"],
                       stoch_completion=st["completion"], stoch_speed=st["mean_speed"])
        res.update(frames=frames_total, final=final, minutes=(time.time() - t0) / 60)
        log["evals"].append(res)
        if verbose:
            print(f"[{name} s{seed}] frames={frames_total:>8} ret={res['mean_return']:7.1f}+-{res['std_return']:5.1f} "
                  f"ema={res.get('ema_return', float('nan')):7.1f} lap={res['completion']:.2f} "
                  f"oof={res['out_of_field']:.2f} cov={res['mean_coverage']:.2f} "
                  f"(med={res['median_coverage']:.2f} p90={res['p90_coverage']:.2f} max={res['max_coverage']:.2f}) "
                  f"off={res['off_road_frac']:.2f} spd={res['mean_speed']:.1f} t90={res['t90']:.0f} "
                  f"| stoch cov={res.get('stoch_coverage', float('nan')):.2f} "
                  f"lap={res.get('stoch_completion', float('nan')):.2f} ({res['minutes']:.1f} min)")
            print("   " + health_report(agent, replay, cfg, log))
        json.dump(log, open(path, "w"))
        if not final:
            if res["completion"] > best_eval["comp"] and res["completion"] > 0:
                best_eval["comp"] = res["completion"]; save_ckpt("best_completion")
            if res["mean_coverage"] > best_eval["cov"]:
                best_eval["cov"] = res["mean_coverage"]; save_ckpt("best_eval_cov")

    while frames_total < cfg.total_frames:
        a = random_action(rng) if agent_steps < cfg.start_steps else agent.act(obs, env.tfrac, deterministic=False)
        obs, r, terminal, ended, info = env.step(a)
        ef.append(info["frame"]); ea.append(a); er.append(r); efail.append(float(info["fail"]))
        ecov.append(info["coverage"])
        frames_total += info["frames"]; agent_steps += 1

        if ended:
            st = info["ep_stats"]
            replay.add(ef, ea, er, efail, terminal, st["coverage"], st["off_road_frac"], st["lap_finished"], ecov)
            log["episodes"].append(dict(st, at_frames=frames_total))
            if st["coverage"] > best_cov:
                best_cov = st["coverage"]
                if best_cov - saved_cov >= 0.02 and (time.time() - t_last_best) > 300:
                    saved_cov = best_cov; t_last_best = time.time()
                    save_ckpt("best_cov")
            if st["lap_finished"]:
                if not os.path.exists(os.path.join(ck_dir, "first_lap.pt")):
                    save_ckpt("first_lap")
                save_ckpt("latest_lap")
            obs, f0 = env.reset()
            ef, ea, er, efail = [f0], [], [], []
            ecov = [_coverage(env.env.unwrapped)]

        if agent_steps >= cfg.start_steps and replay.size >= cfg.batch:
            credit += cfg.updates_per_step
            while credit >= 1.0:
                credit -= 1.0
                agent.update(replay.sample(cfg.batch, cfg.stack, agent.HS_np, agent.seq_len, rng))

        if agent_steps % cfg.log_every == 0 and agent.updates > 0:
            d = agent.pop_metrics(); d.update(frames=frames_total, updates=agent.updates)
            log["diag"].append(d)
        if frames_total >= next_eval:
            do_eval(); next_eval += cfg.eval_every
        if frames_total >= next_ckpt:
            save_ckpt("periodic"); next_ckpt += cfg.ckpt_every

    save_ckpt("final")
    agent.save(os.path.join(out_dir, f"{name}_seed{seed}.pt"))
    do_eval(final=True)
    env.env.close()
    return agent, log
