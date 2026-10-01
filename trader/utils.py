"""Misc helpers: seeding, hardware detection, device plan, model-size tiers."""

import os
import platform

import numpy as np
import torch


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def total_ram_gb():
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return None


def detect_resources():
    """Inventory of the machine: CPUs, RAM and GPUs (with free VRAM)."""
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free_b, total_b = torch.cuda.mem_get_info(i)
            gpus.append(
                {
                    "index": i,
                    "name": props.name,
                    "total_gb": round(total_b / 1e9, 2),
                    "free_gb": round(free_b / 1e9, 2),
                }
            )
    return {
        "hostname": os.uname().nodename if hasattr(os, "uname") else "",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),
        "ram_gb": None if total_ram_gb() is None else round(total_ram_gb(), 2),
        "gpus": gpus,
    }


def format_resources(res) -> str:
    lines = [
        f"host   : {res['hostname']}",
        f"python : {res['python']}  torch={res['torch']}  cuda={res['cuda_available']}",
        f"cpu    : {res['cpu_count']} cores   ram={res['ram_gb']} GB",
    ]
    if res["gpus"]:
        for g in res["gpus"]:
            lines.append(
                f"gpu {g['index']}  : {g['name']}  total={g['total_gb']}GB free={g['free_gb']}GB"
            )
    else:
        lines.append("gpu    : none -> CPU mode")
    return "\n".join(lines)


def size_tier(free_gb, budget_gb):
    """Pick a model-size + intensity tier from the per-GPU VRAM budget
    (capped by free VRAM). Everything can be overridden with CLI flags.

    days_per_epoch = how many random days each epoch trains/rolls out on
    (batched rollouts -> Kx more PPO data per update)."""
    b = min(float(budget_gb), free_gb * 0.85)
    if b >= 16:
        return {"name": "L (16GB+ / A10G)", "pred_hidden": 768, "pred_layers": 2,
                "ana_hidden": 768, "ana_layers": 2,
                "trader_hidden": (8192, 8192, 8192),
                "batch_days": 24, "minibatch": 1024, "days_per_epoch": 8}
    if b >= 8:
        return {"name": "M (8GB+)", "pred_hidden": 256, "pred_layers": 2,
                "ana_hidden": 256, "ana_layers": 2,
                "trader_hidden": (2048, 2048),
                "batch_days": 8, "minibatch": 512, "days_per_epoch": 4}
    if b >= 3:
        return {"name": "S (3GB+)", "pred_hidden": 128, "pred_layers": 1,
                "ana_hidden": 128, "ana_layers": 1,
                "trader_hidden": (1024, 1024),
                "batch_days": 4, "minibatch": 256, "days_per_epoch": 2}
    return {"name": "XS (CPU)", "pred_hidden": 64, "pred_layers": 1,
            "ana_hidden": 64, "ana_layers": 1,
            "trader_hidden": (256, 256),
            "batch_days": 2, "minibatch": 128, "days_per_epoch": 1}


def choose_plan(args, res) -> dict:
    """Devices for the four models + the size/intensity tier.

    4 usable GPUs: predictor=0, analyzer=1, trader A=2, trader B=3
      (predictor/analyzer keep training in background threads on their own
      GPUs while BOTH traders run batched rollouts in parallel).
    3 GPUs: traders A+B share GPU 2.
    2 GPUs: predictor+analyzer share 0, traders share 1.
    1 GPU: everything on it.  0 GPUs (or GPUs too busy, e.g. shared with a
      miner): CPU with the XS tier.

    A GPU is "usable" when it has at least --min-free-gb free VRAM, so we
    never step on other workloads (e.g. a running miner).
    """
    gpus = res["gpus"]
    if args.device == "cpu":
        usable = []
    else:
        usable = [g for g in gpus if g["free_gb"] >= args.min_free_gb]
        if args.device == "cuda" and not usable and gpus:
            usable = gpus  # explicitly forced: ignore the free-memory guard

    def dev(g):
        return torch.device(f"cuda:{g['index']}")

    if len(usable) >= 4:
        plan = {"pred": dev(usable[0]), "ana": dev(usable[1]),
                "trade_a": dev(usable[2]), "trade_b": dev(usable[3]),
                "parallel": True,
                "layout": (f"4+ GPUs: predictor={dev(usable[0])}, analyzer={dev(usable[1])}, "
                           f"traderA={dev(usable[2])}, traderB={dev(usable[3])}")}
    elif len(usable) == 3:
        plan = {"pred": dev(usable[0]), "ana": dev(usable[1]),
                "trade_a": dev(usable[2]), "trade_b": dev(usable[2]),
                "parallel": True,
                "layout": (f"3 GPUs: predictor={dev(usable[0])}, analyzer={dev(usable[1])}, "
                           f"traders A+B={dev(usable[2])}")}
    elif len(usable) == 2:
        plan = {"pred": dev(usable[0]), "ana": dev(usable[0]),
                "trade_a": dev(usable[1]), "trade_b": dev(usable[1]),
                "parallel": True,
                "layout": (f"2 GPUs: predictor+analyzer={dev(usable[0])}, "
                           f"traders A+B={dev(usable[1])}")}
    elif len(usable) == 1:
        plan = {"pred": dev(usable[0]), "ana": dev(usable[0]),
                "trade_a": dev(usable[0]), "trade_b": dev(usable[0]),
                "parallel": False,
                "layout": f"1 GPU: all models on {dev(usable[0])}"}
    else:
        plan = {"pred": torch.device("cpu"), "ana": torch.device("cpu"),
                "trade_a": torch.device("cpu"), "trade_b": torch.device("cpu"),
                "parallel": False,
                "layout": "CPU: all models on cpu"}

    min_free = min((g["free_gb"] for g in usable), default=0.0)
    tier = size_tier(min_free, args.vram_budget_gb)
    plan["tier"] = tier
    return plan
