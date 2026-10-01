# trader_v11 — multi-agent RL trader for BTC / ETH / LTC (1-minute data)

Built for a **4× A10G (23 GB) / 48-core / 120 GB RAM** machine (and auto-scales
down to 1 GPU or CPU). Three agents + **two competing trade makers** trade full
days of 1-minute BTC/ETH/LTC bars, each episode starting with **$20** aiming
for **$30+**.

## What's new in v11 (vs v10)

1. **Uses all the hardware**
   - `--days-per-epoch K` (default **8** on the big tier): each epoch trains
     and trades K random days; all K environments roll out through **one
     batched policy forward per minute** (batched rollouts) → ~11,000 PPO
     transitions per trader per update (8× v10) and real GPU saturation.
   - Predictor/analyzer train on **24-day GRU batches** in one pass.
   - `--preload auto`: the whole 1,338-day dataset is preprocessed into RAM at
     startup with a **multiprocessing pool over all cores** (auto-on with
     ≥32 GB RAM; uses ~1 GB).
   - 4 GPUs → predictor=GPU0, analyzer=GPU1, trader A=GPU2, trader B=GPU3;
     both traders roll out in parallel threads while the predictor/analyzer
     keep training in background threads.
2. **Bigger models on the ≥16 GB tier** (A10G): predictor/analyzer GRU
   **768×2 (~5.4M params each)**, traders MLP **8192-8192-8192 (~147M each)**.
   Peak VRAM is printed every epoch and stays inside `--vram-budget-gb`
   (default 20 GB/GPU) — the batched compute, not idle parameters, is what
   keeps the cards busy.
3. **Per-agent graphs**: every epoch each trader gets its own day-response PNG
   (`epoch_XXXXX_<date>_traderA.png` / `_traderB.png`) and its own curves PNG
   (`training_curves_traderA.png` / `training_curves_traderB.png`), plus a
   shared `training_curves_agents.png` for predictor/analyzer.
4. **Precision**: GAE computed per-trajectory across the K parallel
   environments (exact advantages), advantage normalization over the whole
   K-day batch, batched held-out evaluation.

Kept from v10: vol-normalized 4-horizon targets + anti-flat-line correlation
loss (`pred_corr15/60` — watch it stay off 0), warmup+cosine agent LR,
KL-adaptive trader LR, dual competing traders (best kept per trader).

## Model sizes

| Model | v9 | v10 (A10 tier) | **v11 (A10G tier)** |
|---|---|---|---|
| Price predictor | 17.5K | 2.4M (GRU 512×2) | **5.4M (GRU 768×2 + LN)** |
| Market analyzer | 18.4K | 2.4M | **5.4M** |
| Trade maker ×2 | 1.74M | 113M each | **147M each (8192-8192-8192)** |
| Total | ~1.8M | ~231M | **~305M params** |

## Quick start (Snowflake / any machine)

```bash
pip install -r requirements.txt
python train.py --epochs 2000        # everything auto-detected
```

or `bash scripts/snowflake_run.sh`.

## Outputs (all PNG, every epoch — separate graphs per trading agent)

```
runs/
  epochs/epoch_XXXXX_YYYY-MM-DD_traderA.png   <- trader A's response to that day
  epochs/epoch_XXXXX_YYYY-MM-DD_traderB.png   <- trader B's response to that day
  training_curves_traderA.png                  <- A's own 8-panel board
  training_curves_traderB.png                  <- B's own 8-panel board
  training_curves_agents.png                   <- predictor/analyzer health
  metrics.csv, ckpt/{predictor,analyzer,trader_a,trader_b}.pt (+ best_a/best_b)
  resources.txt, config.json
```

```bash
python eval.py --ckpt runs/ckpt --which both --days 40   # A-vs-B held-out report
```

## Key flags

```
--epochs 2000        --days-per-epoch 8   --vram-budget-gb 20   --preload auto
--device auto        --min-free-gb 3.0    --plot-days 1
--pred-hidden auto   --pred-layers 0      --trader-hidden auto  --minibatch 0
--capital 20         --target 30          --max-lev 5.0         --single-trader
--corr-weight 0.5    --target-kl 0.03     --resume runs/ckpt
```

## Honest notes

* $20 → $30 in one day is a +50 % daily return — extreme; expect a low
  hit-rate and trust the held-out eval columns over training-day equity.
* Fees (0.05 %) + slippage (0.02 %) per rebalance are simulated; churning dies
  to fees, which the agents must learn to avoid.
* Research toy, not financial advice.
