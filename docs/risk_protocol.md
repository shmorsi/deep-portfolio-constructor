# Risk study: protocol

Fixed before any forecast was scored. This file, the code under `src/dpc/risk/`, the forecast
script `scripts/risk_tsfm_forecast.py` and the tests were committed first; the results under
`reports/risk/` were produced afterwards by `python -m dpc.cli risk-study` and committed separately.
Anything that had to change after the first run is listed under [Deviations](#deviations), with
the reason.

## The question

The wealth study asked whether a network could grow a long-only portfolio faster than the
market. This study asks the other half of an investor's problem, risk:

**Does a time-series foundation model, pretrained from scratch on other people's data
([tsfm-from-scratch](https://github.com/shmorsi/tsfm-from-scratch)), forecast next month's
volatility of each US industry better than the standard methods, and does a long-only
portfolio built on its forecasts end up less volatile?**

It is the first place the forecasting project and the investing project meet. Volatility is the
one thing about returns that is known to be forecastable, so it is where a forecasting model
has the best chance of helping, and a loss here is informative too.

## Data

* Kenneth French's 30 US industry portfolios, value-weighted, daily and monthly
  (`30_Industry_Portfolios_daily_CSV.zip`, `30_Industry_Portfolios_CSV.zip`), the wealth study's
  source and vintage. Nothing else.
* **Realized volatility** of industry *i* in month *m*: the square root of the sum of its
  squared daily returns in that month (no mean removed). A month with fewer than 15 daily
  returns for an industry is missing for that industry.
* **Out of sample:** a forecast is made at each month-end from December 1959 to August 2026 for
  the following month, so the months scored are January 1960 to September 2026, the wealth
  study's window. Every forecast uses only data up to its own month-end.

## The forecasters

Each produces, for every industry and month-end, a median and the nine quantiles 0.1 to 0.9 of
next month's realized volatility.

| Name | What it is |
|---|---|
| `last` | Last month's realized volatility (a random walk). |
| `mean60` | The mean of the last 60 months' realized volatility: the long window a covariance estimate usually stands on. |
| `ewma` | RiskMetrics: an exponentially weighted variance of daily returns, decay 0.94 a day, scaled to a 21-day month. |
| `har` | The standard to beat. For each industry, a regression of next month's log volatility on this month's, the mean of the last 3 months and the mean of the last 12 (the monthly form of the HAR model), refit every month on all earlier months. |
| `tsfm_zero` | The pretrained model, zero-shot: it reads the industry's monthly volatility history (as much as its context holds) and forecasts one step. It has never seen these series. |
| `tsfm_ft` | The same model fine-tuned, walk-forward. At the end of December 1959, 1964, ..., 2024 (the wealth study's refit dates) training restarts from the pretrained weights and continues, with the pretraining objective, on the 30 industries' volatility up to ten years before the refit. The ten years before the refit are validation: the step with the lowest pooled quantile loss there is kept, and that may be step 0, the untouched model. AdamW, learning rate 1e-4, batch 64, at most 300 steps, validated every 25 (the settings of tsfm's own volatility protocol). The model fit at a refit forecasts until the next one. |

**Quantiles for the four classical forecasters.** Each gives a point forecast. Its quantile *q*
is that point times the *q*-quantile of its own last 120 forecast errors for that industry
(realized over forecast, known at the forecast date). The classical forecasters therefore start
in December 1949, ten years before the first scored month. The model's quantiles are its own.

**Which model.** Two checkpoints of the same architecture, scored by the same script:

1. `cpu_s0`, seed 0: the 833,440-parameter model whose weights are committed in
   tsfm-from-scratch (`results/runs/cpu_s0_seed0/model.pt`, SHA-256
   `d41b77fd19881097a4d8fc400b4f198a8edbf242a6691147308d5ce2f4a08a91`). Anyone can rerun it.
   On the GIFT-Eval test split this model was worse than AutoARIMA, and that is said here so
   that nobody mistakes it for the strong one. Seeds 1 and 2 are run as a robustness check.
2. `patch_s2`, seed 0: the 10,083,392-parameter headline model of that project (checkpoint
   SHA-256 `85c46256cefa665b7eb0822f9877831563df85e22345670b40c4a115c16d0fd2`). Its weights are
   not public yet, so it is scored when the checkpoint is at hand, by the same script and
   without any change to this protocol. Until then its rows read "not run".

The results for `cpu_s0` will be known before `patch_s2` is scored. Nothing in this file may
change in between; if something has to, it is a deviation.

## What is measured

**Forecast accuracy**, pooled over the 30 industries and the 801 months:

* **Primary: weighted quantile loss** over the levels 0.1 to 0.9 (the sum of the pinball losses
  divided by the sum of the realized volatilities; the "CRPS" of the forecasting project).
* The mean absolute error of the median, QLIKE of the median as a variance forecast, and the
  share of months inside the 10% to 90% band (it should be 80%).
* Each is reported as a **ratio to `har`** with a 95% interval from a moving-block bootstrap
  over months (blocks of 12 months, all industries of a month kept together, 5,000 draws).
  **Better than HAR** needs the whole interval below 1; **worse** needs it above 1; otherwise
  **no detectable difference**.

**What the forecasts are worth to the investor.** One rule, the same for every forecaster:

* At each month-end, the long-only, fully invested portfolio of the 30 industries with the
  smallest forecast variance, at most 20% in one industry (the wealth study's cap). The
  covariance is the forecaster's volatilities combined with one correlation matrix, the same
  for all of them: the Ledoit-Wolf estimate on the last 60 monthly returns.
* Traded monthly through the wealth study's simulator, 10 basis points per unit traded.
* **Primary: the portfolio's realized variance** each month, from its daily returns with the
  weights left to drift inside the month. A forecaster's portfolio is compared with `har`'s by
  the mean difference in log realized variance over the 801 months, with the wealth study's
  test: Newey-West *t* (12 lags) and a 95% block-bootstrap interval (12-month blocks, 5,000
  draws). **Lower risk** needs *t* ≤ −2 and the whole interval below zero; **higher risk** the
  reverse; otherwise **no detectable difference**.
* Also reported: yearly volatility, growth of $1 after costs, worst fall and trading of each
  portfolio; the same comparison for a second rule (weights proportional to one over the
  forecast volatility); and two reference portfolios that are not forecasts: the market, and
  an **oracle** that is handed next month's realized volatilities. The oracle cannot be
  invested in. It shows how much a perfect forecast would be worth to this rule, which bounds
  what any forecaster can add.

## What would count as the model earning its place

For a checkpoint, the claim "its risk forecasts help this investor" needs **both**: `tsfm_ft`
better than `har` on the weighted quantile loss, **and** its minimum-variance portfolio at
lower risk than `har`'s by the test above. Anything else is reported as no evidence that the
foundation model's risk forecasts help. `tsfm_zero` is reported beside it and is not the
primary variant.

Stated in advance: I expect `cpu_s0` to lose to `har`, because a regression fitted to each
industry's own history is a hard baseline for a small model that has never seen it. For
`patch_s2` I do not know.

## What is reported whatever happens

Every forecaster's scores and every portfolio's numbers; the scores by decade; the three seeds
of `cpu_s0`; how many fine-tuning refits kept the untouched model; and the oracle's gap, even
if it shows the question has little room in it.

## Limits fixed in advance

* A month's realized volatility from about 21 daily returns is a noisy measurement of risk.
* Thirty industries, one country. Not individual stocks, not other asset classes.
* The portfolio rule is one rule. A different use of the same forecasts (position sizing,
  leverage, options) is not tested and nothing is claimed about it.
* The fine-tuned model is refit every five years, the regression every month. That is how each
  is normally used, and it favours the regression.
* One training seed of each checkpoint is the primary run.

## Deviations

None.

## Added after the first run

Dated 2026-11-11, after the scores above were known. Nothing pre-registered changed.

* **A descriptive comparison.** The protocol compares every forecaster with `har`. The results
  also show the fine-tuned model against each other forecaster (last month, the 60-month
  mean, RiskMetrics, and its own zero-shot run), by the same loss and the same bootstrap. It
  is labelled "described after the test" wherever it appears and does not enter the decision
  rule.
* **A note on the two risk figures.** The tables show each portfolio's yearly realized
  volatility (the root of the average monthly variance) beside the pre-registered comparison
  (the average difference in log variance). The documents now say which is which.
