# Deep Portfolio Constructor

**A market-neutral long-short equity portfolio that keeps the intended stock bets and removes unintended sector and factor risk, using a transformer covariance model and an end-to-end differentiable optimiser. It is tested honestly against classical portfolio construction.**

## Summary

I built a research pipeline that turns three textbook signals (12-1 momentum, short-term reversal and low volatility) into a dollar-neutral portfolio of 99 US and Canadian large caps. Every long-short strategy goes through one shared convex optimiser, which makes the portfolio neutral to sectors, market beta, size and value. Only the inputs differ:
* a classical Ledoit-Wolf covariance
* a deep covariance model (patch transformer over time plus self-attention across stocks, with a factor-structured, always-positive-definite output)
* an end-to-end model trained *through* the optimiser with cvxpylayers, on realised risk-adjusted return after costs

Everything is walk-forward and out-of-sample from January 2008 to September 2026, with yearly refits and explicit look-ahead checks.

**Headline findings:**
* **The constrained optimiser does most of the work.** It raises the share of risk coming from intended stock bets from **17% to 61%**.
* **The deep covariance model** forecasts risk better (best out-of-sample likelihood) but does not hedge better.
* **The end-to-end model** has the best net Sharpe (**0.58 vs −0.03**; 0.51 ± 0.06 across three seeds), but the edge is not statistically significant (t = 1.7). About a third of it is market beta that leaked through the beta constraint.

## Results

Out-of-sample, January 2008 to September 2026, monthly rebalancing, net of 10 bps per unit traded:

| Strategy | Net Sharpe | Sortino | Max DD | Ann. turnover | FF6 alpha (t) | % risk from intended bets |
|---|---|---|---|---|---|---|
| Equal weight (long-only) | 0.72 | 1.02 | −48.5% | 0.6× | 2.9% (3.83)* | 1% |
| Min variance (long-only) | 0.60 | 0.83 | −38.1% | 3.6× | −0.6% (−0.42) | 8% |
| Mean-variance, Ledoit-Wolf (long-only) | 0.60 | 0.84 | −36.9% | 11.7× | −0.7% (−0.58) | 10% |
| Risk parity (long-only) | 0.72 | 1.01 | −45.0% | 0.7× | 2.0% (2.42)* | 2% |
| HRP (long-only) | 0.72 | 1.01 | −41.3% | 3.1× | 1.8% (2.05)* | 2% |
| L/S naive signal | −0.40 | −0.54 | −87.7% | 23.7× | −7.8% (−3.47) | 17% |
| L/S classical (LW + optimiser) | −0.03 | −0.05 | −31.9% | 12.0× | −1.2% (−0.69) | **61%** |
| L/S deep risk + optimiser | −0.05 | −0.07 | −36.3% | 12.0× | −2.0% (−1.24) | 41% |
| **L/S end-to-end deep** | **0.58** | **0.84** | **−27.6%** | 11.1× | 3.1% (1.48) | 47% |

\* Long-only alphas are inflated by survivorship bias (see Limitations). "% risk from intended bets" is the ex-ante stock-specific share of variance under an *independent*, evaluation-only Barra-style model (6 style factors and 11 sectors).

**Deep risk model vs classical**, out-of-sample at monthly rebalance dates:

| | Gaussian NLL (lower is better) | Months best | Min-variance realised vol |
|---|---|---|---|
| Sample covariance | −2.690 | 0% | 14.7% |
| Ledoit-Wolf | −2.899 | 48% | **13.3%** |
| **Deep (transformer)** | **−2.941** | **51%** | 14.0% |

**End-to-end vs classical:**
* Difference in monthly net returns: +0.43% a month, Newey-West t = 1.70, over 225 months.
* After hedging each strategy's own trailing beta (estimated point-in-time, so the comparison starts in 2009), the end-to-end Sharpe falls from 0.64 to 0.40 on that same window. About 37% of the edge is leaked market beta.
* Across seeds 42 / 43 / 44, the end-to-end Sharpe is 0.58 / 0.46 / 0.50. Across risk-model sizes of 8K–109K parameters, the deep model always wins on likelihood and always loses to Ledoit-Wolf on min-variance volatility.

Full tables, crisis years (2008, 2020, 2022) and sensitivity tests are in [`reports/stage6_evaluation.md`](reports/stage6_evaluation.md). The 2-page write-up is [`reports/research_note.md`](reports/research_note.md).

![Risk decomposition](reports/figures/stage6_risk_decomposition.png)

## How to run

Requires Python 3.11 or later and about 6 GB of disk for PyTorch. The full pipeline takes about 1.5 hours on a 4-core CPU, and a GPU isn't needed.

Set up the environment:

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

Then run the stages one at a time:

```bash
python -m dpc.cli data          # Stage 1: download, quality-check, build the panel
```

```bash
python -m dpc.cli signals       # Stage 2: signals and information coefficients
```

```bash
python -m dpc.cli baselines     # Stage 3: classical baselines on the shared engine
```

```bash
python -m dpc.cli risk          # Stage 4: walk-forward deep covariance model (~15 min)
```

```bash
python -m dpc.cli e2e           # Stage 5: end-to-end differentiable optimiser (~10 min)
```

```bash
python -m dpc.cli robustness    # Stage 6a: model-size and seed reruns (~1 h)
```

```bash
python -m dpc.cli evaluate      # Stage 6b: metrics, risk decomposition, sensitivity, look-ahead checks
```

```bash
streamlit run dashboard/app.py  # Stage 7: dashboard
```

```bash
pytest                          # 77 tests
```

`python -m dpc.cli all` runs Stages 1–6 in order. Every setting lives in [`config.yaml`](config.yaml); the loader rejects unknown or missing keys.

On a laptop without an NVIDIA GPU, you can install the much smaller CPU build of PyTorch first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Project layout

```
config.yaml               every setting (strictly validated)
data/universe.csv         tickers, country, GICS sector
src/dpc/
  data/                   download (yfinance, Ken French), quality checks, clean panel
  signals.py              momentum, reversal, low-vol; IC with Newey-West t-stats
  exposures.py            point-in-time rolling factor betas
  covariance.py           sample and Ledoit-Wolf
  optimizer.py            the ONE long-short problem (cvxpy, DPP-compliant)
  backtest.py             walk-forward engine; PointInTimeView hides the future
  baselines.py            EW, min-var, MV-LW, risk parity, HRP, naive and classical L/S
  deep/features.py        point-in-time tensors
  deep/risk_model.py      patch transformer + cross-sectional attention -> D(BB'+Psi)D
  deep/train_risk.py      walk-forward training, embargo, out-of-sample evaluation
  deep/e2e.py             end-to-end model through a cvxpylayers optimiser layer
  evaluation.py           metrics, Barra-style risk decomposition, tests, look-ahead checks
  robustness.py           model-size and seed reruns
dashboard/app.py          Streamlit dashboard
tests/                    pytest suite (engine, constraints, gradients, look-ahead)
reports/                  research note, evaluation tables, figures
```

## Look-ahead safeguards

* **Point-in-time access.** Strategies only see a `PointInTimeView` that slices data at the decision date. The engine raises an error on any weight in a non-investable stock.
* **Truncation invariance.** Tests and a runtime check on the real data scramble every return after a date T and assert that all decisions up to T are unchanged. This covers the signals, eligibility, backtest weights and deep-model inputs.
* **Walk-forward deep training.** Every training label must end on or before the refit date, and there is an embargo between training and validation windows. Both are asserted in code.
* **Oracle calibration.** A deliberately cheating strategy that trades on next month's returns scores a Sharpe of 10.6, which shows what look-ahead would look like.

## Limitations

* ⚠️ **Survivorship bias.** No point-in-time index membership was available (no WRDS/CRSP). The universe is *today's* large caps, so delisted and fallen companies are missing. This inflates long-only returns and alphas and penalises the low-volatility signal. Sector labels are also current GICS, not point-in-time.
* **Returns-based exposures.** Size and value exposure are rolling betas to SMB and HML, not market-cap or book-to-market characteristics. Beta estimation error let the end-to-end model carry a realised market beta of about 0.19 despite an ex-ante limit of ±0.05.
* **Not statistically significant.** The end-to-end improvement is not significant at 5%, and several variants were tried during development, so a multiple-testing haircut would be needed.
* **Simplified execution.** Trades fill at the monthly close at a flat 10 bps, with no borrow costs, market impact or capacity limits.
* **Data truncation.** The sample ends in September 2026 because the Ken French factors are published with a lag.
* **Partial retraining in sensitivity tests.** Constraint and cost sensitivity re-uses the trained forecasts, and model-size sensitivity retrains only the risk model.
