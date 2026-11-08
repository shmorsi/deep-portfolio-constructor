# Deep Portfolio Constructor

> **Status:** Stage 0 (setup) complete. Results, the results table and the limitations
> section are filled in as later stages are built. Nothing below is a result yet.

## Summary

This project builds a long-short equity portfolio that keeps the stock bets we
*intend* to make (from simple, well-known signals) while removing risk we did
*not* intend to take (sector, market beta, size and value exposure). It uses a
deep-learning model to forecast the covariance matrix and trains the whole
pipeline end to end through a convex optimizer (cvxpylayers). Then it tests
honestly whether this beats classical portfolio construction. The headline
metric is the **share of portfolio risk that comes from intended stock bets
versus unintended factor and sector exposure**.

## Results

_To be filled in after Stage 6._

| Strategy | Net Sharpe | Sortino | Max DD | Ann. turnover | Factor alpha (t) | % risk from intended bets |
|---|---|---|---|---|---|---|
| Equal weight | | | | | | |
| Min variance | | | | | | |
| Mean-variance (Ledoit-Wolf) | | | | | | |
| Risk parity | | | | | | |
| HRP | | | | | | |
| Deep + differentiable optimizer | | | | | | |

## Project layout

```
config.yaml          every setting (seeds, paths, and per-stage parameters)
requirements.txt     pinned dependencies
pyproject.toml       package metadata + pytest settings
data/raw/            downloaded data cache (git-ignored)
data/processed/      cleaned panels (git-ignored)
src/dpc/             the library
  config.py          strict, typed loader for config.yaml
  reproducibility.py seeds, deterministic torch, run manifest
  logging_utils.py   one log format everywhere
  setup_check.py     Stage 0 smoke test (incl. cvxpylayers gradient check)
notebooks/           exploration only. Logic lives in src/
tests/               pytest suite
reports/             research note, figures, run manifest
```

## How to run

Requires Python 3.11 or later (developed on 3.13).

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
pip install -e .
```

```bash
python -m dpc.setup_check
```

```bash
pytest
```

PyTorch: `requirements.txt` installs the default PyPI wheel. On a laptop without
an NVIDIA GPU, you can install the much smaller CPU build first with
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Reproducibility

* One master seed (`project.seed` in `config.yaml`) drives numpy, torch and Python's `random`.
* Torch runs with deterministic kernels and a fixed thread count.
* `python -m dpc.setup_check` writes `reports/run_manifest.json` with the seed, git
  commit and every library version, so any number in the report can be traced to an
  exact environment.

## Roadmap

| Stage | Content | Status |
|---|---|---|
| 0 | Setup, config, reproducibility | ✅ |
| 1 | Data pipeline (prices, Fama-French, sectors, quality checks) | ⏳ |
| 2 | Signals: momentum, reversal, low-vol. IC and t-stats | ⏳ |
| 3 | Classical baselines plus a shared walk-forward backtest engine | ⏳ |
| 4 | Deep covariance model vs. sample and Ledoit-Wolf | ⏳ |
| 5 | End-to-end differentiable optimizer with neutrality constraints | ⏳ |
| 6 | Rigorous evaluation, risk decomposition, crisis periods, sensitivity | ⏳ |
| 7 | Streamlit dashboard and research note | ⏳ |

## Limitations

_To be completed. This section will include a **survivorship bias** warning if
historical index membership data is not available (see Stage 1)._
