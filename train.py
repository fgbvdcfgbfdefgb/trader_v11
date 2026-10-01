#!/usr/bin/env python3
"""
trader_v11 - offline multi-agent RL trading training, tuned for 4x A10G
(23GB) / 48-core / 120GB-RAM machines - and it still auto-scales down to a
single GPU or CPU.

What changed vs v10
-------------------
* USE ALL THE HARDWARE:
  - `--days-per-epoch K` (default 8 on the big tier): every epoch trains and
    trades K random days, and all K environments roll out through ONE batched
    policy forward per minute (batched rollouts) -> Kx more PPO data per
    update (~11k transitions/trader/update at K=8) and far better GPU
    utilization. GAE is computed per-trajectory so the advantages stay exact.
  - predictor/analyzer train on batches of up to 24 days in one GRU pass.
  - `--preload auto` (default ON with >=32GB RAM): the whole dataset is
    preprocessed into RAM at startup using a multiprocessing pool over all CPU
    cores -> zero per-epoch data-prep latency.
  - With 4 GPUs: predictor=GPU0, analyzer=GPU1, trader A=GPU2, trader B=GPU3,
    both traders roll out in parallel threads while the predictor/analyzer
    keep training in background threads on their own GPUs.
* BIGGER MODELS on the >=16GB tier: predictor/analyzer GRU 768x2 (~5.4M
  params each), traders MLP 8192-8192-8192 (~147M params each) - sized to
  stay inside the --vram-budget-gb cap (default 20GB/GPU) while the batched
  compute actually keeps the cards busy.
* PER-AGENT GRAPHS: every epoch each trader gets its OWN day-response PNG
  (epoch_XXXXX_<date>_traderA.png / _traderB.png) and its OWN training
  curves PNG (training_curves_traderA/B.png); predictor/analyzer share
  training_curves_agents.png.
* Kept from v10: vol-normalized 4-horizon targets + anti-flat-line
  correlation loss, warmup+cosine agent LR schedule, KL-adaptive trader LR,
  dual competing traders, held-out evaluation.

Fully offline (dataset ships in the repo).
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import threading
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trader import data as D  # noqa: E402
from trader import plots  # noqa: E402
from trader.agents import Analyzer, Predictor, Trader  # noqa: E402
from trader.data import PAIRS  # noqa: E402
from trader.env import TradingEnv  # noqa: E402
from trader.features import (  # noqa: E402
    HORIZONS,
    IDX15,
    IDX60,
    N_FEATURES,
    W,
    build_features,
    build_labels,
    build_targets,
)
from trader.ppo import RolloutBuffer, ppo_update_buffers  # noqa: E402
from trader.utils import choose_plan, detect_resources, format_resources, set_seed  # noqa: E402

EMB_DIM = 16
CSV_FIELDS = [
    "epoch", "days",
    "pred_loss", "pred_mae15", "pred_mae60", "pred_corr15", "pred_corr60", "lr_pred",
    "ana_loss", "ana_vol_acc", "ana_trend_acc", "lr_ana",
    "pi_loss_a", "v_loss_a", "entropy_a", "kl_a", "lr_trader_a",
    "ep_ret_a", "final_eq_a", "best_eq_a", "hit_a", "margin_call_a",
    "pi_loss_b", "v_loss_b", "entropy_b", "kl_b", "lr_trader_b",
    "ep_ret_b", "final_eq_b", "best_eq_b", "hit_b", "margin_call_b",
    "eval_mean_eq_a", "eval_median_eq_a", "eval_hit_a",
    "eval_mean_eq_b", "eval_median_eq_b", "eval_hit_b",
    "elapsed",
]


class DayProcessed:
    __slots__ = ("date", "day", "feats", "Y_pct", "Y_norm", "M", "vol_scale",
                 "vol_lab", "trend_lab", "lab_mask")

    def __init__(self, date, day):
        self.date = date
        self.day = day
        self.feats = build_features(day)
        self.Y_pct, self.Y_norm, self.M, self.vol_scale = build_targets(day)
        self.vol_lab, self.trend_lab, self.lab_mask = build_labels(day)


def _build_proc(args):
    date, day = args
    return DayProcessed(date, day)


def _corr_np(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


class GlobalSched:
    """Warmup + cosine-decay LR schedule over the whole run, stepped per
    optimizer update (one thread touches a given optimizer at a time)."""

    def __init__(self, opt, base_lr, total_steps, warmup=100, floor=0.05):
        self.opt = opt
        self.base = base_lr
        self.total = max(int(total_steps), 1)
        self.warmup = warmup
        self.floor = floor
        self.i = 0

    def step(self):
        if self.i < self.warmup:
            f = (self.i + 1) / self.warmup
        else:
            t = min((self.i - self.warmup) / max(self.total - self.warmup, 1), 1.0)
            f = self.floor + (1 - self.floor) * 0.5 * (1.0 + math.cos(math.pi * t))
        lr = self.base * f
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.i += 1
        return lr


def parse_args():
    p = argparse.ArgumentParser(
        description="trader_v11 multi-agent RL training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", default="./data")
    p.add_argument("--out", default="./runs")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    # hardware / sizes
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--min-free-gb", type=float, default=3.0)
    p.add_argument("--vram-budget-gb", type=float, default=20.0)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--pred-hidden", default="auto")
    p.add_argument("--pred-layers", type=int, default=0)
    p.add_argument("--ana-hidden", default="auto")
    p.add_argument("--ana-layers", type=int, default=0)
    p.add_argument("--trader-hidden", default="auto", help="'auto' or comma list")
    p.add_argument("--batch-days", type=int, default=0)
    p.add_argument("--minibatch", type=int, default=0)
    p.add_argument("--days-per-epoch", type=int, default=0,
                   help="random days per epoch (0 = tier default; 8 on the big tier)")
    p.add_argument("--max-days", type=int, default=0)
    p.add_argument("--preload", choices=["auto", "yes", "no"], default="auto",
                   help="preprocess the whole dataset into RAM at startup "
                        "(multiprocessing pool over all cores; auto = on with >=32GB RAM)")
    # second trader
    p.add_argument("--single-trader", action="store_true")
    p.add_argument("--seed-b", type=int, default=1042)
    p.add_argument("--ent-b", type=float, default=0.008)
    p.add_argument("--lr-b", type=float, default=5e-4)
    # data split
    p.add_argument("--val-days", type=int, default=40)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-limit", type=int, default=16)
    # market / task
    p.add_argument("--capital", type=float, default=20.0)
    p.add_argument("--target", type=float, default=30.0)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--max-lev", type=float, default=5.0)
    p.add_argument("--terminal-bonus", type=float, default=0.5)
    # predictor / analyzer
    p.add_argument("--pred-steps", type=int, default=8)
    p.add_argument("--ana-steps", type=int, default=8)
    p.add_argument("--replay", type=int, default=48)
    p.add_argument("--corr-weight", type=float, default=0.5)
    p.add_argument("--bg-steps", type=int, default=4)
    p.add_argument("--no-background-a", action="store_true")
    # ppo
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--lr-trader", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.003)
    p.add_argument("--lr-agents", type=float, default=1e-3, dest="lr_agents_base")
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--target-kl", type=float, default=0.03)
    # logging
    p.add_argument("--plot-every", type=int, default=1)
    p.add_argument("--plot-days", type=int, default=1,
                   help="how many of the epoch's days get per-trader PNGs")
    p.add_argument("--ckpt-every", type=int, default=10)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--resume", default="")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "epochs"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "ckpt"), exist_ok=True)
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    # ---------------- resources & plan ----------------
    res = detect_resources()
    plan = choose_plan(args, res)
    tier = plan["tier"]

    pred_hidden = tier["pred_hidden"] if args.pred_hidden == "auto" else int(args.pred_hidden)
    pred_layers = tier["pred_layers"] if args.pred_layers <= 0 else args.pred_layers
    ana_hidden = tier["ana_hidden"] if args.ana_hidden == "auto" else int(args.ana_hidden)
    ana_layers = tier["ana_layers"] if args.ana_layers <= 0 else args.ana_layers
    trader_hidden = list(tier["trader_hidden"]) if args.trader_hidden == "auto" \
        else [int(h) for h in str(args.trader_hidden).split(",")]
    batch_days = tier["batch_days"] if args.batch_days <= 0 else args.batch_days
    minibatch = tier["minibatch"] if args.minibatch <= 0 else args.minibatch
    K = tier["days_per_epoch"] if args.days_per_epoch <= 0 else args.days_per_epoch

    print(format_resources(res))
    print(plan["layout"])
    print(f"size tier: {tier['name']} | predictor GRU h={pred_hidden} x{pred_layers} | "
          f"analyzer GRU h={ana_hidden} x{ana_layers} | trader MLP {trader_hidden} | "
          f"batch_days={batch_days} minibatch={minibatch} days/epoch={K}")
    with open(os.path.join(args.out, "resources.txt"), "w") as f:
        f.write(format_resources(res) + "\n\n" + plan["layout"] + "\n"
                + f"tier={tier['name']} pred h={pred_hidden}x{pred_layers} "
                f"ana h={ana_hidden}x{ana_layers} trader {trader_hidden} "
                f"days/epoch={K}\n")
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({**vars(args), "plan": {k: str(v) for k, v in plan.items()},
                   "tier": tier}, f, indent=2)

    set_seed(args.seed)

    # ---------------- data ----------------
    days = D.load_days(args.data, max_days=args.max_days)
    dates = sorted(days)
    if len(dates) < 20:
        raise RuntimeError("need at least 20 valid days of data")
    n_val = args.val_days
    if n_val >= len(dates) - 10:
        n_val = max(0, (len(dates) - 10) // 2)
    val_dates = dates[len(dates) - n_val:] if n_val > 0 else []
    train_dates = dates[: len(dates) - n_val]
    print(f"[data] train days: {len(train_dates)} ({train_dates[0]}..{train_dates[-1]})"
          f" | held-out val days: {len(val_dates)}")

    # ---------------- preload (use all cores / RAM) ----------------
    preload = args.preload == "yes" or (args.preload == "auto" and (res["ram_gb"] or 0) >= 32)
    procs_all = {}
    if preload:
        t0 = time.time()
        try:
            from multiprocessing import Pool
            workers = min(32, res["cpu_count"] or 4)
            with Pool(workers) as pool:
                built = pool.map(_build_proc, [(d, days[d]) for d in dates], chunksize=16)
            procs_all = {p.date: p for p in built}
            print(f"[data] preloaded {len(procs_all)} days into RAM "
                  f"({workers} workers, {time.time() - t0:.1f}s)")
        except Exception as e:  # noqa: BLE001
            print(f"[data] preload failed ({e}); falling back to lazy processing")
            procs_all = {}

    # ---------------- day cache / replay ----------------
    proc_cache = OrderedDict()
    cache_lock = threading.Lock()
    cache_cap = max(args.replay, K * 4, 8)

    def get_processed(date):
        if procs_all:
            return procs_all[date]
        with cache_lock:
            if date in proc_cache:
                proc_cache.move_to_end(date)
                return proc_cache[date]
        proc = DayProcessed(date, days[date])
        with cache_lock:
            proc_cache[date] = proc
            while len(proc_cache) > cache_cap:
                proc_cache.popitem(last=False)
        return proc

    def sample_batch(today=None, weight_today=4):
        with cache_lock:
            pool = list(procs_all.values()) if procs_all else list(proc_cache.values())
        if today is not None:
            pool = list(today) * weight_today + pool
        if not pool:
            return None
        n = min(batch_days, len(pool))
        return random.sample(pool, n)

    # ---------------- agents ----------------
    dev_p, dev_a = plan["pred"], plan["ana"]
    dev_ta, dev_tb = plan["trade_a"], plan["trade_b"]
    n_out = len(PAIRS) * len(HORIZONS)
    obs_dim = W * N_FEATURES + n_out + EMB_DIM + 3 + 1

    predictor = Predictor(N_FEATURES, n_out, hidden=pred_hidden, layers=pred_layers).to(dev_p)
    analyzer = Analyzer(N_FEATURES, len(PAIRS), hidden=ana_hidden, layers=ana_layers,
                        emb_dim=EMB_DIM).to(dev_a)
    torch.manual_seed(args.seed)
    trader_a = Trader(obs_dim, act_dim=3, hidden=trader_hidden).to(dev_ta)
    torch.manual_seed(args.seed_b)
    trader_b = Trader(obs_dim, act_dim=3, hidden=trader_hidden).to(dev_tb)
    set_seed(args.seed)

    opt_p = torch.optim.AdamW(predictor.parameters(), lr=args.lr_agents_base, weight_decay=1e-5)
    opt_a = torch.optim.AdamW(analyzer.parameters(), lr=args.lr_agents_base, weight_decay=1e-5)
    opt_ta = torch.optim.Adam(trader_a.parameters(), lr=args.lr_trader)
    opt_tb = torch.optim.Adam(trader_b.parameters(), lr=args.lr_b)

    steps_per_epoch = args.pred_steps + (args.bg_steps if (plan["parallel"] and not args.no_background_a) else 0)
    total_agent_steps = max(1, args.epochs) * steps_per_epoch + 200
    sched_p = GlobalSched(opt_p, args.lr_agents_base, total_agent_steps)
    sched_a = GlobalSched(opt_a, args.lr_agents_base, total_agent_steps)

    for name, m in (("predictor", predictor), ("analyzer", analyzer),
                    ("trader A", trader_a), ("trader B", trader_b)):
        n_par = sum(p.numel() for p in m.parameters())
        print(f"[agents] {name:10s}: {n_par/1e6:.2f}M params ({n_par*4/1e6:.1f} MB fp32)")
    print(f"[agents] trader obs_dim={obs_dim} | batched rollouts over {K} days/epoch | "
          f"two traders: {'A+B' if not args.single_trader else 'A only'}")

    # ---------------- resume ----------------
    start_epoch = 1
    mult_a, mult_b = 1.0, 1.0
    if args.resume:
        for name, model, opt in (
            ("predictor", predictor, opt_p),
            ("analyzer", analyzer, opt_a),
            ("trader_a", trader_a, opt_ta),
            ("trader_b", trader_b, opt_tb),
        ):
            pth = os.path.join(args.resume, f"{name}.pt")
            if os.path.exists(pth):
                ck = torch.load(pth, map_location="cpu", weights_only=False)
                model.load_state_dict(ck["model"])
                opt.load_state_dict(ck["opt"])
                start_epoch = max(start_epoch, int(ck.get("epoch", 0)) + 1)
        for name, sched in (("predictor", sched_p), ("analyzer", sched_a)):
            pth = os.path.join(args.resume, f"{name}.pt")
            if os.path.exists(pth):
                ck = torch.load(pth, map_location="cpu", weights_only=False)
                sched.i = int(ck.get("sched_i", 0))
        for name, opt, base in (("trader_a", opt_ta, args.lr_trader), ("trader_b", opt_tb, args.lr_b)):
            pth = os.path.join(args.resume, f"{name}.pt")
            if os.path.exists(pth):
                ck = torch.load(pth, map_location="cpu", weights_only=False)
                mult = float(ck.get("lr_mult", 1.0))
                if name == "trader_a":
                    mult_a = mult
                else:
                    mult_b = mult
                for g in opt.param_groups:
                    g["lr"] = base * mult
        print(f"[resume] continuing from epoch {start_epoch}")

    # ---------------- agent training steps ----------------
    def predictor_train_batch(batch):
        x = torch.as_tensor(np.stack([d.feats for d in batch]), device=dev_p)
        Yn = torch.as_tensor(np.stack([d.Y_norm for d in batch]), device=dev_p)
        M = torch.as_tensor(np.stack([d.M for d in batch]), device=dev_p)
        pred = predictor(x)
        mse = ((pred - Yn).pow(2) * M).sum() / M.sum().clamp(min=1.0)
        p_ = torch.where(M, pred, torch.zeros_like(pred))
        y_ = torch.where(M, Yn, torch.zeros_like(Yn))
        n = M.sum(dim=(0, 1)).clamp(min=2)
        pm = p_.sum(dim=(0, 1)) / n
        ym = y_.sum(dim=(0, 1)) / n
        cov = (p_ * y_).sum(dim=(0, 1)) / n - pm * ym
        vp = (p_ * p_).sum(dim=(0, 1)) / n - pm.pow(2)
        vy = (y_ * y_).sum(dim=(0, 1)) / n - ym.pow(2)
        corr = cov / (vp.clamp_min(0).sqrt() * vy.clamp_min(0).sqrt() + 1e-6)
        corr = corr.clamp(-1.0, 1.0)
        loss = mse + args.corr_weight * (1.0 - corr.mean())
        sched_p.step()
        opt_p.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        opt_p.step()
        return float(loss.item()), float(corr.mean().item())

    def analyzer_train_batch(batch):
        x = torch.as_tensor(np.stack([d.feats for d in batch]), device=dev_a)
        vol = torch.as_tensor(np.stack([d.vol_lab for d in batch]), device=dev_a)
        tr = torch.as_tensor(np.stack([d.trend_lab for d in batch]), device=dev_a)
        mask = torch.as_tensor(np.stack([d.lab_mask for d in batch]), device=dev_a)
        emb, vl, tl = analyzer(x)
        B, T = mask.shape
        v = vl.reshape(B * T, len(PAIRS), 3)
        t_ = tl.reshape(B * T, len(PAIRS), 3)
        yv = vol.reshape(B * T, len(PAIRS))
        yt = tr.reshape(B * T, len(PAIRS))
        m = mask.reshape(B * T)
        loss = F.cross_entropy(v[m], yv[m]) + F.cross_entropy(t_[m], yt[m])
        sched_a.step()
        opt_a.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(analyzer.parameters(), 1.0)
        opt_a.step()
        return float(loss.item())

    @torch.no_grad()
    def predictor_eval(procs):
        """Averaged metrics across the epoch's K days (pct space)."""
        x = torch.as_tensor(np.stack([p.feats for p in procs]), device=dev_p)
        Yn = torch.as_tensor(np.stack([p.Y_norm for p in procs]), device=dev_p)
        M = torch.as_tensor(np.stack([p.M for p in procs]), device=dev_p)
        pred = predictor(x)
        out = {"loss": float(((pred - Yn).pow(2) * M).sum().item() / max(M.sum().item(), 1.0))}
        pred_np = pred.cpu().numpy()
        for tag, k in (("15", IDX15), ("60", IDX60)):
            maes, corrs = [], []
            for b, p in enumerate(procs):
                for i in range(len(PAIRS)):
                    c = i * len(HORIZONS) + k
                    mc = p.M[:, c]
                    if mc.sum() <= 30:
                        continue
                    pp = (pred_np[b, :, c] * p.vol_scale[:, c])[mc]
                    yv = p.Y_pct[:, c][mc]
                    maes.append(float(np.abs(pp - yv).mean()))
                    corrs.append(_corr_np(pp, yv))
            out[f"mae{tag}"] = float(np.mean(maes)) if maes else float("nan")
            out[f"corr{tag}"] = float(np.mean(corrs)) if corrs else float("nan")
        return out

    @torch.no_grad()
    def analyzer_eval(procs):
        x = torch.as_tensor(np.stack([p.feats for p in procs]), device=dev_a)
        vol = torch.as_tensor(np.stack([p.vol_lab for p in procs]), device=dev_a)
        tr = torch.as_tensor(np.stack([p.trend_lab for p in procs]), device=dev_a)
        mask = torch.as_tensor(np.stack([p.lab_mask for p in procs]), device=dev_a)
        emb, vl, tl = analyzer(x)
        m = mask.reshape(-1).cpu().numpy()
        v = vl.reshape(-1, len(PAIRS), 3).argmax(-1).cpu().numpy()
        t_ = tl.reshape(-1, len(PAIRS), 3).argmax(-1).cpu().numpy()
        yv = tr.reshape(-1, len(PAIRS)).cpu().numpy()
        yt = vol.reshape(-1, len(PAIRS)).cpu().numpy()
        va = float((v[m] == yv[m]).mean()) if m.any() else float("nan")
        ta = float((t_[m] == yt[m]).mean()) if m.any() else float("nan")
        vv = vl.reshape(-1, len(PAIRS), 3)
        tt = tl.reshape(-1, len(PAIRS), 3)
        yv2 = tr.reshape(-1, len(PAIRS))
        yt2 = vol.reshape(-1, len(PAIRS))
        mm = mask.reshape(-1)
        loss = float(F.cross_entropy(vv[mm], yv2[mm]).item()
                     + F.cross_entropy(tt[mm], yt2[mm]).item())
        return {"loss": loss, "vol_acc": va, "trend_acc": ta}

    @torch.no_grad()
    def advisor_outputs_batch(procs):
        """(K,T,n_out) normalized forecasts, % forecasts, embeddings, vol classes."""
        x = torch.as_tensor(np.stack([p.feats for p in procs]), device=dev_p)
        fn = predictor(x).cpu().numpy().astype(np.float32)
        xa = x.to(dev_a)
        emb, vl, tl = analyzer(xa)
        emb = emb.cpu().numpy().astype(np.float32)
        vol_cls = vl.argmax(-1).cpu().numpy().astype(np.int64)  # (K,T,3)
        fp = fn * np.stack([p.vol_scale for p in procs])
        return fn, fp, emb, vol_cls

    # ---------------- trading ----------------
    env_kwargs = dict(capital=args.capital, target=args.target, fee=args.fee,
                      slip=args.slip, max_lev=args.max_lev,
                      terminal_bonus=args.terminal_bonus)

    def batched_rollout(trader, procs, fn, emb, bufs=None, deterministic=False):
        """Roll out ONE environment per day, all stepping through a single
        batched policy forward per minute. GAE buffers stay per-trajectory."""
        envs = [TradingEnv(p.day, p.feats, fn[i], emb[i], **env_kwargs)
                for i, p in enumerate(procs)]
        E = len(envs)
        obs = np.stack([e.reset() for e in envs])
        active = np.ones(E, dtype=bool)
        finals = np.zeros(E, dtype=np.float64)
        ep_rets = np.zeros(E, dtype=np.float64)
        if not deterministic and bufs is not None:
            for b in bufs:
                b.clear()
        dev = next(trader.parameters()).device
        while bool(active.any()):
            o_t = torch.as_tensor(obs, device=dev)
            with torch.no_grad():
                if deterministic:
                    mu, _, _ = trader(o_t)
                    a_np = mu.clamp(-1.0, 1.0).cpu().numpy()
                else:
                    mu, std, v = trader(o_t)
                    dist = Normal(mu, std)
                    raw = dist.sample()
                    logp = dist.log_prob(raw).sum(-1)
                    a_np = raw.clamp(-1.0, 1.0).cpu().numpy()
                    logp_np = logp.cpu().numpy()
                    v_np = v.cpu().numpy()
            for i in np.nonzero(active)[0]:
                prev = obs[i].copy()
                o2, r, d, info = envs[i].step(a_np[i])
                obs[i] = o2
                if not deterministic and bufs is not None:
                    bufs[i].add(prev, a_np[i], float(logp_np[i]), float(v_np[i]), r, d)
                    ep_rets[i] += r
                if d:
                    active[i] = False
                    finals[i] = info["equity"]
        return envs, finals, ep_rets

    def evaluate(trader, limit):
        sel = list(val_dates)
        step = max(1, len(sel) // max(limit, 1))
        sel = sel[::step][:limit]
        finals = []
        for s in range(0, len(sel), K):
            chunk = sel[s : s + K]
            procs = [procs_all[d] if procs_all else DayProcessed(d, days[d]) for d in chunk]
            fn, _, emb, _ = advisor_outputs_batch(procs)
            _, fin, _ = batched_rollout(trader, procs, fn, emb, deterministic=True)
            finals.extend(list(fin))
        finals = np.asarray(finals, dtype=np.float64)
        return {"mean": float(finals.mean()), "median": float(np.median(finals)),
                "hit": float((finals >= args.target).mean())}

    def adapt_lr(opt, base_lr, mult, kl):
        if not math.isfinite(kl):
            return mult
        if kl > 1.5 * args.target_kl:
            mult *= 0.7
        elif kl < args.target_kl / 1.5:
            mult *= 1.3
        mult = min(max(mult, 0.05), 20.0)
        for g in opt.param_groups:
            g["lr"] = base_lr * mult
        return mult

    def save_ckpt(directory, epoch):
        os.makedirs(directory, exist_ok=True)
        for name, model, opt, extra in (
            ("predictor", predictor, opt_p, {"sched_i": sched_p.i}),
            ("analyzer", analyzer, opt_a, {"sched_i": sched_a.i}),
            ("trader_a", trader_a, opt_ta, {"lr_mult": mult_a}),
            ("trader_b", trader_b, opt_tb, {"lr_mult": mult_b}),
        ):
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "epoch": epoch, **extra},
                       os.path.join(directory, f"{name}.pt"))

    # ---------------- metrics history ----------------
    hist = {k: [] for k in CSV_FIELDS}
    csv_path = os.path.join(args.out, "metrics.csv")
    if args.resume and os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                for k in CSV_FIELDS:
                    v = row.get(k, "")
                    if k == "date":
                        hist[k].append(v)
                    else:
                        try:
                            hist[k].append(float(v))
                        except (TypeError, ValueError):
                            hist[k].append(float("nan"))

    csv_new = not os.path.exists(csv_path)
    csv_f = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
    if csv_new:
        writer.writeheader()

    used_devices = sorted({dev_p, dev_a, dev_ta, dev_tb}, key=str)
    vram_peak = {}

    # ---------------- main loop ----------------
    best_eval = {"a": -1e9, "b": -1e9}
    bufs_a = [RolloutBuffer() for _ in range(K)]
    bufs_b = [RolloutBuffer() for _ in range(K)]
    t_start = time.time()
    epoch = start_epoch - 1
    print(f"[train] start: {args.epochs} epochs x {K} days/epoch | "
          f"${args.capital} -> ${args.target} | fee={args.fee} slip={args.slip} "
          f"max_lev={args.max_lev} | traders A+"
          + ("" if not args.single_trader else " (B disabled)"))

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            ep_dates = random.sample(train_dates, min(K, len(train_dates)))
            procs = [get_processed(d) for d in ep_dates]

            # ---- Phase A: predictor + analyzer train on the epoch's days (+pool)
            def _pa():
                for _ in range(args.pred_steps):
                    b = sample_batch(today=procs)
                    if b:
                        predictor_train_batch(b)

            def _aa():
                for _ in range(args.ana_steps):
                    b = sample_batch(today=procs)
                    if b:
                        analyzer_train_batch(b)

            ths = [threading.Thread(target=_pa), threading.Thread(target=_aa)]
            for t in ths:
                t.start()
            for t in ths:
                t.join()

            pm = predictor_eval(procs)
            am = analyzer_eval(procs)

            # advisor outputs for all the epoch's days (frozen for the rollouts)
            fn, fp, emb, vol_cls = advisor_outputs_batch(procs)

            # ---- Phase B: both traders run batched rollouts over the K days;
            #      predictor/analyzer keep training in background threads.
            bg = []
            if plan["parallel"] and not args.no_background_a:
                def _bgp():
                    for _ in range(args.bg_steps):
                        b = sample_batch()
                        if b:
                            predictor_train_batch(b)

                def _bga():
                    for _ in range(args.bg_steps):
                        b = sample_batch()
                        if b:
                            analyzer_train_batch(b)

                bg = [threading.Thread(target=_bgp), threading.Thread(target=_bga)]
                for t in bg:
                    t.start()

            res_a, res_b = {}, {}

            def _roll_a():
                res_a["envs"], res_a["finals"], res_a["rets"] = batched_rollout(
                    trader_a, procs, fn, emb, bufs_a)

            def _roll_b():
                res_b["envs"], res_b["finals"], res_b["rets"] = batched_rollout(
                    trader_b, procs, fn, emb, bufs_b)

            if plan["parallel"] and not args.single_trader:
                rths = [threading.Thread(target=_roll_a), threading.Thread(target=_roll_b)]
                for t in rths:
                    t.start()
                for t in rths:
                    t.join()
            else:
                _roll_a()
                if not args.single_trader:
                    _roll_b()

            ppo_a = ppo_update_buffers(trader_a, opt_ta, bufs_a, dev_ta, gamma=args.gamma,
                                       lam=args.lam, ppo_epochs=args.ppo_epochs,
                                       minibatch=minibatch, ent_coef=args.ent_coef,
                                       target_kl=args.target_kl)
            mult_a = adapt_lr(opt_ta, args.lr_trader, mult_a, ppo_a.get("kl", float("nan")))
            if not args.single_trader:
                ppo_b = ppo_update_buffers(trader_b, opt_tb, bufs_b, dev_tb, gamma=args.gamma,
                                           lam=args.lam, ppo_epochs=args.ppo_epochs,
                                           minibatch=minibatch, ent_coef=args.ent_b,
                                           target_kl=args.target_kl)
                mult_b = adapt_lr(opt_tb, args.lr_b, mult_b, ppo_b.get("kl", float("nan")))
            else:
                ppo_b = {k: float("nan") for k in ("pi_loss", "v_loss", "entropy", "kl", "clip_frac")}

            for t in bg:
                t.join()

            fin_a = np.asarray(res_a["finals"])
            hit_a = float((fin_a >= args.target).mean())
            if not args.single_trader:
                fin_b = np.asarray(res_b["finals"])
                hit_b = float((fin_b >= args.target).mean())
            else:
                fin_b = np.array([float("nan")])
                hit_b = float("nan")

            # ---- held-out evaluation (both traders)
            eval_row = {k: float("nan") for k in
                        ("eval_mean_eq_a", "eval_median_eq_a", "eval_hit_a",
                         "eval_mean_eq_b", "eval_median_eq_b", "eval_hit_b")}
            if val_dates and (epoch % args.eval_every == 0 or epoch == args.epochs):
                ev_a = evaluate(trader_a, args.eval_limit)
                eval_row.update({"eval_mean_eq_a": ev_a["mean"],
                                 "eval_median_eq_a": ev_a["median"],
                                 "eval_hit_a": ev_a["hit"]})
                if ev_a["mean"] > best_eval["a"]:
                    best_eval["a"] = ev_a["mean"]
                    save_ckpt(os.path.join(args.out, "ckpt", "best_a"), epoch)
                if not args.single_trader:
                    ev_b = evaluate(trader_b, args.eval_limit)
                    eval_row.update({"eval_mean_eq_b": ev_b["mean"],
                                     "eval_median_eq_b": ev_b["median"],
                                     "eval_hit_b": ev_b["hit"]})
                    if ev_b["mean"] > best_eval["b"]:
                        best_eval["b"] = ev_b["mean"]
                        save_ckpt(os.path.join(args.out, "ckpt", "best_b"), epoch)

            # ---- VRAM accounting
            for d in used_devices:
                if d.type == "cuda":
                    try:
                        vram_peak[str(d)] = max(vram_peak.get(str(d), 0.0),
                                                torch.cuda.max_memory_allocated(d) / 1e9)
                    except RuntimeError:
                        pass

            # ---- logs / per-agent plots / checkpoints
            row = {
                "epoch": epoch, "days": len(procs),
                "pred_loss": pm["loss"], "pred_mae15": pm["mae15"], "pred_mae60": pm["mae60"],
                "pred_corr15": pm["corr15"], "pred_corr60": pm["corr60"],
                "lr_pred": opt_p.param_groups[0]["lr"],
                "ana_loss": am["loss"], "ana_vol_acc": am["vol_acc"],
                "ana_trend_acc": am["trend_acc"], "lr_ana": opt_a.param_groups[0]["lr"],
                "pi_loss_a": ppo_a["pi_loss"], "v_loss_a": ppo_a["v_loss"],
                "entropy_a": ppo_a["entropy"], "kl_a": ppo_a["kl"],
                "lr_trader_a": opt_ta.param_groups[0]["lr"],
                "ep_ret_a": float(np.mean(res_a["rets"])), "final_eq_a": float(fin_a.mean()),
                "best_eq_a": float(fin_a.max()), "hit_a": hit_a,
                "margin_call_a": float(np.mean([e.margin_call for e in res_a["envs"]])),
                "pi_loss_b": ppo_b["pi_loss"], "v_loss_b": ppo_b["v_loss"],
                "entropy_b": ppo_b["entropy"], "kl_b": ppo_b["kl"],
                "lr_trader_b": opt_tb.param_groups[0]["lr"] if not args.single_trader else float("nan"),
                "ep_ret_b": float(np.mean(res_b["rets"])) if not args.single_trader else float("nan"),
                "final_eq_b": float(fin_b.mean()) if not args.single_trader else float("nan"),
                "best_eq_b": float(fin_b.max()) if not args.single_trader else float("nan"),
                "hit_b": hit_b,
                "margin_call_b": float(np.mean([e.margin_call for e in res_b["envs"]]))
                if not args.single_trader else float("nan"),
                **eval_row, "elapsed": round(time.time() - t0, 2),
            }
            writer.writerow(row)
            csv_f.flush()
            for k in CSV_FIELDS:
                hist[k].append(row.get(k, float("nan")))

            if args.plot_every and (epoch % args.plot_every == 0 or epoch == args.epochs):
                btc15 = 0 * len(HORIZONS) + IDX15
                for pd_i in range(min(args.plot_days, len(procs))):
                    closes = np.stack([procs[pd_i].day[p]["close"] for p in PAIRS], axis=1)
                    common = dict(epoch=epoch, date=ep_dates[pd_i], closes=closes,
                                  capital=args.capital, target=args.target,
                                  forecast15=fp[pd_i, :, btc15],
                                  realized15=procs[pd_i].Y_pct[:, btc15],
                                  corr15=pm["corr15"], vol_classes=vol_cls[pd_i])
                    plots.plot_trader_day(
                        os.path.join(args.out, "epochs",
                                     f"epoch_{epoch:05d}_{ep_dates[pd_i]}_traderA.png"),
                        tag="a", weights=res_a["envs"][pd_i].weights_hist,
                        equity=res_a["envs"][pd_i].equity_hist,
                        hit=bool(fin_a[pd_i] >= args.target),
                        margin_call=bool(res_a["envs"][pd_i].margin_call),
                        extra=f"day {pd_i + 1}/{len(procs)}", **common)
                    if not args.single_trader:
                        plots.plot_trader_day(
                            os.path.join(args.out, "epochs",
                                         f"epoch_{epoch:05d}_{ep_dates[pd_i]}_traderB.png"),
                            tag="b", weights=res_b["envs"][pd_i].weights_hist,
                            equity=res_b["envs"][pd_i].equity_hist,
                            hit=bool(fin_b[pd_i] >= args.target),
                            margin_call=bool(res_b["envs"][pd_i].margin_call),
                            extra=f"day {pd_i + 1}/{len(procs)}", **common)
                plots.plot_curves_trader(
                    os.path.join(args.out, "training_curves_traderA.png"),
                    hist, "a", args.capital, args.target)
                if not args.single_trader:
                    plots.plot_curves_trader(
                        os.path.join(args.out, "training_curves_traderB.png"),
                        hist, "b", args.capital, args.target)
                plots.plot_curves_agents(
                    os.path.join(args.out, "training_curves_agents.png"), hist)

            if epoch % args.ckpt_every == 0 or epoch == args.epochs:
                save_ckpt(os.path.join(args.out, "ckpt"), epoch)

            if args.log_every and (epoch % args.log_every == 0 or epoch == args.epochs):
                ev_txt = ""
                if not math.isnan(eval_row["eval_mean_eq_a"]):
                    ev_txt = (f" | EVAL A ${eval_row['eval_mean_eq_a']:.2f}"
                              f"({eval_row['eval_hit_a']*100:.0f}%)"
                              + (f" B ${eval_row['eval_mean_eq_b']:.2f}"
                                 f"({eval_row['eval_hit_b']*100:.0f}%)"
                                 if not math.isnan(eval_row["eval_mean_eq_b"]) else ""))
                vram_txt = " ".join(f"{k}={v:.1f}GB" for k, v in sorted(vram_peak.items()))
                print(f"[{epoch:5d}] x{len(procs)}d A:${fin_a.mean():6.2f}(best ${fin_a.max():6.2f}, {hit_a*100:.0f}% hit)"
                      + (f" B:${fin_b.mean():6.2f}(best ${fin_b.max():6.2f}, {hit_b*100:.0f}% hit)"
                         if not args.single_trader else "")
                      + f" pCorr15={pm['corr15']:+.3f} aTrd={am['trend_acc']:.2f}"
                      f" lrP={opt_p.param_groups[0]['lr']:.1e}"
                      f" lrA={opt_ta.param_groups[0]['lr']:.1e}"
                      + (f" lrB={opt_tb.param_groups[0]['lr']:.1e}" if not args.single_trader else "")
                      + f"{ev_txt} | {vram_txt} [{time.time() - t0:.1f}s]", flush=True)
    finally:
        csv_f.close()
        save_ckpt(os.path.join(args.out, "ckpt"), epoch)
        print(f"[train] finished {epoch} epochs in {(time.time() - t_start)/60:.1f} min; "
              f"artifacts in {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
