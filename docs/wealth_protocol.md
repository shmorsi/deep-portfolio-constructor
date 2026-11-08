# Pre-registered protocol: the long-only wealth study

Committed before the Deep Growth Network was run out of sample. Every number
the study reports is produced by this protocol; nothing below is changed after
seeing out-of-sample results. If a bug is found later, the fix and its effect
are reported next to the result, not folded in silently.

## Question

Can a neural network, trained on the growth of wealth after trading costs,
build a better long-only, fully invested, long-term portfolio than simply
holding the market, and than the classical rules, from 1960 to 2026?

## Data and universe

* Kenneth French's 30 US industry portfolios, value-weighted monthly total
  returns, July 1926 to September 2026 (CRSP 202609 vintage). Each portfolio
  holds every listed firm in the industry at each date, including firms that
  later failed: no survivorship bias.
* Industry market caps (number of firms times average firm size) give the
  market portfolio. Timing is validated in `reports/wealth/data_summary.json`:
  the cap-weighted portfolio of the 30 industries tracks French's market return
  with correlation 0.9998.
* Book-to-market for year Y (formed at the end of June of Y) is used from the
  end of July of Y. The one-month T-bill rate and the Fama-French factors come
  from the same library.

## Investing rules (all strategies)

* Long-only, fully invested in equities, no leverage, no shorting, no cash timing.
* A decision at the end of month t is held through month t+1. Holdings drift
  with prices between trades. Every unit traded costs 10 bps (one way).
* Out of sample: January 1960 to September 2026 (801 months). Every strategy
  starts from cash in December 1959 and pays to buy its first portfolio.

## The model (primary, pre-specified: `deep`)

* Deep Growth Network: 13 point-in-time features per industry plus where the
  current holding sits relative to the market, a market context token, two
  transformer blocks (d_model 32, 4 heads) across the 30 industries.
* Target weights: market weights tilted by at most 3x either way; the tilt head
  starts at zero, so the untrained network holds the market. A learned trade
  fraction moves holdings part of the way to the target each month.
* Trained through a differentiable simulator of the investing rules above, on
  minus the average monthly log growth of wealth after costs (Kelly; risk
  aversion 1), in 60-month episodes.
* Walk-forward refits every 5 years at the end of December (1959, 1964, ...,
  2024). Inside each refit, the last 10 years are validation (early stopping on
  validation growth), the rest training; no training label is dated after the
  first validation decision (asserted in code and tested). Seeds 42 to 46, each
  warm-started from its own previous refit; the deployed policy averages them.

## Comparisons

| Strategy | Rule |
|---|---|
| `market` | Hold every industry at its market-cap weight |
| `equal_weight` | 1/30 each, rebalanced every January |
| `min_variance` | Lowest-risk long-only mix, at most 20% per industry, Ledoit-Wolf covariance of the last 60 months, every January |
| `risk_parity` | Equal risk from every industry, every January |
| `growth_classical` | Same goal and mandate as the network (maximise mu'w - w'Sigma w / 2 within 1/3x..3x of market weights) with plug-in estimates: long-run means shrunk halfway to their average, Ledoit-Wolf covariance; every January |
| `momentum_tilt` | Market weights tilted by 12-1 month momentum rank, within the same 3x bounds, monthly |

These classical strategies were run once while the simulator was being built,
with the parameters above fixed beforehand; none was changed afterwards.

## Primary comparison and decision rule

* Primary metric: annualised difference in net log growth, `deep` minus
  `market`, January 1960 to September 2026.
* "Better" if the Newey-West t-statistic (12 lags) is at least 2 and the 95%
  moving-block bootstrap interval (12-month blocks, 5,000 draws) excludes zero;
  "worse" if both point the other way; otherwise "no detectable difference".
* Secondary comparisons, with the same rule, reported but not used to choose
  anything: `deep` against `equal_weight`, `momentum_tilt` and
  `growth_classical`.

## Also reported

Terminal wealth of $1, CAGR, volatility, Sharpe ratio, maximum drawdown,
turnover and average holding period, CAGR by decade, the share of rolling
10-year windows in which each strategy beat the market, Fama-French three-factor
alpha with Newey-West t, each seed's policy on its own, costs of 0, 10, 25 and
50 bps (same decisions re-priced), and three ablations fixed in advance:
`deep_trade_all` (no learned holding), `deep_mlp` (no attention) and
`deep_unanchored` (no market anchor), each with seeds 42 and 43.
