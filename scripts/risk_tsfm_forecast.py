#!/usr/bin/env python3
"""The foundation model's volatility forecasts for the risk study (docs/risk_protocol.md).

This script runs in tsfm-from-scratch's environment: it imports tsfm, numpy and torch, and
nothing from this repository. It reads the realized-volatility panel written by
`python -m dpc.cli risk-data` and writes two forecast files in the format of
dpc.risk.forecasters:

    reports/risk/forecasts/tsfm_zero_<name>.npz   the pretrained model, zero-shot
    reports/risk/forecasts/tsfm_ft_<name>.npz     fine-tuned, walk-forward (refit every five years)

Run it from a checkout of tsfm-from-scratch:

    uv run python ../deep-portfolio-constructor/scripts/risk_tsfm_forecast.py \\
        --checkpoint results/runs/cpu_s0_seed0/model.pt --name cpu_s0

Everything below the imports is fixed by the protocol. A checkpoint the protocol names
(cpu_s0, patch_s2) is refused unless its SHA-256 is the one written there.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

DPC = Path(__file__).resolve().parents[1]
LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
OOS_START = "1960-01"          # first month scored; the first forecast is made at the end of 1959-12
REFIT_EVERY_YEARS = 5          # fine-tuning restarts at the end of December 1959, 1964, ..., the wealth study's refit dates
VAL_MONTHS = 120               # the ten years before a refit choose the step that is kept
STEPS, EVAL_EVERY = 300, 25    # tsfm's own volatility protocol: at most 300 steps, validated every 25
LEARNING_RATE, BATCH = 1e-4, 64
BETAS, WEIGHT_DECAY, GRAD_CLIP = (0.9, 0.95), 0.0, 1.0
MIN_CONTEXT, MIN_END = 32, 64
MAX_CONTEXT = 2048
SEED = 0
FLOOR = 1e-6                   # a volatility quantile at or below zero is set to this, and counted
PROTOCOL_SHA256 = {
    "cpu_s0": "d41b77fd19881097a4d8fc400b4f198a8edbf242a6691147308d5ce2f4a08a91",
    "patch_s2": "85c46256cefa665b7eb0822f9877831563df85e22345670b40c4a115c16d0fd2",
}


def load_panel(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return [str(d) for d in z["dates"]], [str(n) for n in z["names"]], z["rv"].astype(np.float64)


def origin_indices(dates: list[str]) -> np.ndarray:
    """Decision indices scored: the month before OOS_START up to the second-to-last month."""
    return np.arange(dates.index(OOS_START) - 1, len(dates) - 1)


def refit_indices(dates: list[str], idx: np.ndarray) -> list[int]:
    """The scored decisions at which fine-tuning restarts: every fifth December from the first one."""
    first_year = int(dates[idx[0]][:4])
    return [int(t) for t in idx if dates[t].endswith("-12") and (int(dates[t][:4]) - first_year) % REFIT_EVERY_YEARS == 0]


def history(rv: np.ndarray, t: int) -> list[np.ndarray]:
    """Each industry's realized volatility up to and including month t, from its first observed month."""
    out = []
    for i in range(rv.shape[1]):
        s = rv[:t + 1, i]
        seen = np.flatnonzero(np.isfinite(s))
        out.append(s[seen[0]:].astype(np.float32) if seen.size else s[:0].astype(np.float32))
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pooled_wql(y: np.ndarray, q: np.ndarray) -> float:
    """Pooled weighted quantile loss over LEVELS. y [M, N], q [M, N, Q]."""
    d = y[:, :, None] - q
    lv = np.asarray(LEVELS)[None, None, :]
    return float(2.0 * np.maximum(lv * d, (lv - 1.0) * d).sum() / len(LEVELS) / np.abs(y).sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path, help="model.pt or ckpt.pt of a tsfm run")
    ap.add_argument("--name", required=True, help="cpu_s0, cpu_s0_seed1, cpu_s0_seed2 or patch_s2")
    ap.add_argument("--panel", type=Path, default=DPC / "reports" / "risk" / "panel.npz")
    ap.add_argument("--out", type=Path, default=DPC / "reports" / "risk" / "forecasts")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    import torch
    from tsfm.forecast import forecast, load_model
    from tsfm.train.objective import compute_loss, draw_context

    digest = sha256_file(a.checkpoint)
    if a.name in PROTOCOL_SHA256 and digest != PROTOCOL_SHA256[a.name]:
        raise SystemExit(f"{a.checkpoint} is not the {a.name} checkpoint of the protocol (SHA-256 {digest})")
    for variant in ("zero", "ft"):
        target = a.out / f"tsfm_{variant}_{a.name}.npz"
        if target.exists():
            raise SystemExit(f"{target} already exists. The protocol scores a checkpoint once; delete nothing, log a deviation instead.")

    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else "cpu" if a.device == "auto" else a.device)
    dates, names, rv = load_panel(a.panel)
    idx = origin_indices(dates)
    refits = refit_indices(dates, idx)
    pretrained = load_model(str(a.checkpoint), device)
    columns = [pretrained.cfg.quantiles.index(lv) for lv in LEVELS]
    floored = {"zero": 0, "ft": 0}

    def predict(model, ts: list[int], variant: str | None = None) -> np.ndarray:
        """[len(ts), N, Q]: one-step quantiles for each decision month in ts."""
        model.eval()
        out = np.empty((len(ts), len(names), len(LEVELS)))
        with torch.no_grad():
            for k, t in enumerate(ts):
                out[k] = forecast(model, history(rv, t), 1, MAX_CONTEXT).quantiles[:, 0, columns]
        if variant is not None:
            floored[variant] += int((out <= FLOOR).sum())
        return np.maximum(out, FLOOR)

    def val_score(model, refit: int) -> float:
        ts = list(range(refit - VAL_MONTHS, refit))                     # their targets are months refit-119 .. refit
        return pooled_wql(rv[np.asarray(ts) + 1], predict(model, ts))

    def finetune(refit: int):
        """Continue the pretraining objective on volatility up to ten years before the refit; keep the best validation step."""
        model = copy.deepcopy(pretrained)
        window = min(1024, model.cfg.patch_len * model.cfg.max_seq_len) if model.cfg.tokenizer == "patch" else min(1024, model.cfg.max_seq_len + 1)
        train = [s for s in history(rv, refit - VAL_MONTHS) if s.size >= MIN_END]
        rng = np.random.default_rng(SEED)
        torch.manual_seed(SEED)
        opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=BETAS, weight_decay=WEIGHT_DECAY)
        best = val_score(model, refit)
        best_state, kept, log = copy.deepcopy(model.state_dict()), 0, {"0": best}
        for step in range(1, STEPS + 1):
            model.train()
            x = np.full((BATCH, window), np.nan, np.float32)
            for b in range(BATCH):
                s = train[int(rng.integers(len(train)))]
                end = int(rng.integers(MIN_END, s.size + 1))
                seg = s[max(0, end - window):end]
                x[b, window - seg.size:] = seg
            ctx = draw_context(rng, x, MIN_CONTEXT, window - 1)
            loss, _ = compute_loss(model, torch.from_numpy(x).to(device), torch.from_numpy(ctx).to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            if step % EVAL_EVERY == 0:
                v = val_score(model, refit)
                log[str(step)] = v
                if v < best:
                    best, kept, best_state = v, step, copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        return model, {"refit": dates[refit], "train_until": dates[refit - VAL_MONTHS], "kept_step": kept, "val_wql": log}

    started = time.time()
    q_zero = predict(pretrained, [int(t) for t in idx], "zero")
    q_ft = np.empty_like(q_zero)
    refit_log = []
    for k, refit in enumerate(refits):
        model, entry = finetune(refit)
        until = refits[k + 1] if k + 1 < len(refits) else int(idx[-1]) + 1
        rows = [int(t) for t in idx if refit <= t < until]
        q_ft[np.asarray(rows) - int(idx[0])] = predict(model, rows, "ft")
        refit_log.append(entry)
        print(f"refit {entry['refit']}: kept step {entry['kept_step']}, validation loss {entry['val_wql'][str(entry['kept_step'])]:.5f} (step 0: {entry['val_wql']['0']:.5f})", flush=True)

    tsfm_commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "unknown"
    meta = {"kind": "tsfm", "checkpoint": a.name, "checkpoint_sha256": digest, "parameters": int(sum(p.numel() for p in pretrained.parameters())),
            "tsfm_commit": tsfm_commit, "torch": torch.__version__, "numpy": np.__version__, "device": str(device), "seed": SEED,
            "max_context": MAX_CONTEXT, "seconds": round(time.time() - started, 1)}
    origins = [dates[t] for t in idx]
    a.out.mkdir(parents=True, exist_ok=True)
    for variant, q, extra in (("zero", q_zero, {"floored": floored["zero"]}), ("ft", q_ft, {"floored": floored["ft"], "refits": refit_log})):
        np.savez_compressed(a.out / f"tsfm_{variant}_{a.name}.npz", q=q.astype(np.float32), levels=np.array(LEVELS), origins=np.array(origins),
                            names=np.array(names), meta=np.array(json.dumps({**meta, "variant": variant, **extra}, sort_keys=True)))
        print(f"wrote {a.out / f'tsfm_{variant}_{a.name}.npz'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
