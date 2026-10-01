#!/usr/bin/env python3
"""Evaluate saved checkpoints on the most recent (held-out) days.

Usage:
  python eval.py --ckpt runs/ckpt --days 40              # trader A
  python eval.py --ckpt runs/ckpt --which both --days 40 # A vs B competition
Produces per-trader CSVs, per-day PNGs (one per trader) and eval_summary.png.
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from trader import data as D  # noqa: E402
from trader import plots  # noqa: E402
from trader.agents import Analyzer, Predictor, Trader  # noqa: E402
from trader.data import PAIRS  # noqa: E402
from trader.env import TradingEnv  # noqa: E402
from trader.features import HORIZONS, IDX15, N_FEATURES  # noqa: E402
from train import EMB_DIM, DayProcessed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/ckpt")
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="runs/eval")
    ap.add_argument("--which", choices=["a", "b", "both"], default="a")
    ap.add_argument("--days", type=int, default=40)
    ap.add_argument("--capital", type=float, default=20.0)
    ap.add_argument("--target", type=float, default=30.0)
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--slip", type=float, default=0.0002)
    ap.add_argument("--max-lev", type=float, default=5.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    days = D.load_days(args.data)
    dates = sorted(days)[-args.days:]
    print(f"[eval] {len(dates)} days on {device}")

    n_out = len(PAIRS) * len(HORIZONS)
    ck_p = torch.load(os.path.join(args.ckpt, "predictor.pt"),
                      map_location=device, weights_only=False)
    pred_hid = ck_p["model"]["gru.weight_hh_l0"].shape[0] // 3
    pred_layers = sum(1 for k in ck_p["model"] if k.startswith("gru.weight_ih_l"))
    predictor = Predictor(N_FEATURES, n_out, hidden=pred_hid, layers=pred_layers).to(device)
    predictor.load_state_dict(ck_p["model"])
    predictor.eval()

    ck_a = torch.load(os.path.join(args.ckpt, "analyzer.pt"),
                      map_location=device, weights_only=False)
    ana_hid = ck_a["model"]["gru.weight_hh_l0"].shape[0] // 3
    ana_layers = sum(1 for k in ck_a["model"] if k.startswith("gru.weight_ih_l"))
    analyzer = Analyzer(N_FEATURES, len(PAIRS), hidden=ana_hid, layers=ana_layers,
                        emb_dim=EMB_DIM).to(device)
    analyzer.load_state_dict(ck_a["model"])
    analyzer.eval()

    traders = {}
    tags = ["a"] if args.which == "a" else ["b"] if args.which == "b" else ["a", "b"]
    for tag in tags:
        ck_t = torch.load(os.path.join(args.ckpt, f"trader_{tag}.pt"),
                          map_location=device, weights_only=False)
        obs_dim = ck_t["model"]["trunk.0.weight"].shape[1]
        widths = [v.shape[0] for k, v in ck_t["model"].items()
                  if k.startswith("trunk.") and k.endswith(".weight")]
        traders[tag] = Trader(obs_dim, act_dim=3, hidden=widths).to(device)
        traders[tag].load_state_dict(ck_t["model"])
        traders[tag].eval()

    env_kwargs = dict(capital=args.capital, target=args.target, fee=args.fee,
                      slip=args.slip, max_lev=args.max_lev)

    all_rows = {tag: [] for tag in traders}
    for date in dates:
        proc = DayProcessed(date, days[date])
        with torch.no_grad():
            x = torch.as_tensor(proc.feats[None], device=device)
            fn = predictor(x)[0].cpu().numpy().astype(np.float32)
            fp = fn * proc.vol_scale
            emb, vl, _ = analyzer(x)
            emb = emb[0].cpu().numpy().astype(np.float32)
            vol_cls = vl[0].argmax(-1).cpu().numpy()
        for tag, trader in traders.items():
            env = TradingEnv(proc.day, proc.feats, fn, emb, **env_kwargs)
            obs = env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    mu, _, _ = trader(torch.as_tensor(obs, device=device))
                    a = mu.clamp(-1, 1).cpu().numpy()
                obs, r, done, info = env.step(a)
            all_rows[tag].append({"date": date, "final_eq": info["equity"],
                                  "hit": float(info["equity"] >= args.target),
                                  "margin_call": float(env.margin_call)})
            print(f"  [{date}] trader {tag.upper()}: ${info['equity']:7.2f} "
                  f"{'HIT' if info['equity'] >= args.target else '   '}")
            # per-trader day PNG
            closes = np.stack([proc.day[p]["close"] for p in PAIRS], axis=1)
            plots.plot_trader_day(
                os.path.join(args.out, f"eval_{date}_trader{tag.upper()}.png"),
                0, date, tag, closes, env.weights_hist, env.equity_hist,
                args.capital, args.target, fp[:, IDX15], proc.Y_pct[:, IDX15],
                0.0, vol_cls, hit=bool(info["equity"] >= args.target),
                margin_call=bool(env.margin_call), extra="held-out evaluation")

    for tag, rows in all_rows.items():
        with open(os.path.join(args.out, f"eval_results_{tag}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["date", "final_eq", "hit", "margin_call"])
            w.writeheader()
            w.writerows(rows)

    fig, ax = plt.subplots(figsize=(15, 6), dpi=110)
    width = 0.8 / len(all_rows)
    for j, (tag, rows) in enumerate(sorted(all_rows.items())):
        finals = np.array([r["final_eq"] for r in rows])
        hits = np.array([r["hit"] for r in rows])
        x = np.arange(len(rows)) + (j - len(all_rows) / 2 + 0.5) * width
        ax.bar(x, finals, width=width,
               color=plots.TRADER_COLORS.get(tag, "#8e44ad"),
               label=f"trader {tag.upper()}: mean ${finals.mean():.2f}, "
                     f"hit {hits.mean()*100:.1f}%")
    ax.axhline(args.capital, color="gray", ls="--", lw=1, label=f"start ${args.capital:.0f}")
    ax.axhline(args.target, color="#2ecc71", ls=":", lw=1.4, label=f"target ${args.target:.0f}")
    ax.set_xticks(np.arange(len(dates)), [d[5:] for d in dates], rotation=90, fontsize=7)
    ax.set_ylabel("final equity ($)")
    ax.set_title(f"Held-out evaluation: {len(dates)} days | "
                 + " vs ".join(f"{t.upper()} mean ${np.mean([r['final_eq'] for r in rows]):.2f}"
                               for t, rows in sorted(all_rows.items())))
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "eval_summary.png"))
    plt.close(fig)
    for tag, rows in sorted(all_rows.items()):
        finals = np.array([r["final_eq"] for r in rows])
        hits = np.array([r["hit"] for r in rows])
        print(f"[eval] trader {tag.upper()}: mean ${finals.mean():.2f} "
              f"median ${np.median(finals):.2f} hit {hits.mean()*100:.1f}% "
              f"max ${finals.max():.2f}")
    print(f"[eval] artifacts -> {args.out}")


if __name__ == "__main__":
    main()
