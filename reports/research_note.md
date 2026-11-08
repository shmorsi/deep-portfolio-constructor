# Deep Portfolio Construction: Can a Learned Risk Model and a Differentiable Optimiser Keep the Bets We Want and Drop the Ones We Don't?

**Author:** Sherif · **Date:** November 2026 · **Universe:** 99 US & Canadian large caps · **Sample:** Jan 2008 – Sep 2026, out-of-sample, monthly, net of 10 bps

---

## Bottom line

* **Portfolio construction matters more than the model.** The same three signals traded naively lose 8.7% a year, with 83% of their risk coming from factor and sector exposure nobody intended. Put through a constrained optimiser (dollar-, sector-, beta-, size- and value-neutral), the share of risk from intended stock bets rises from **17% to 61%** and volatility falls from 18% to 8%.
* **The deep risk model is a better forecaster but not a better hedger.** A transformer factor-covariance model beats Ledoit-Wolf on out-of-sample likelihood (wins 51% of months). But it produces a *higher*-volatility minimum-variance portfolio (14.0% vs 13.3%) and a two-stage strategy that is no better than classical (Sharpe −0.05 vs −0.03).
* **End-to-end training delivers the best net Sharpe (0.58 for the main seed, 0.51 ± 0.06 across three seeds, vs −0.03 for classical), but it is not statistically conclusive and is partly leaked beta.** The edge over classical is +0.43% a month (Newey-West t = 1.70). The model learned to press against its +0.05 beta limit, and realised beta was 0.19. Hedging its own trailing beta point-in-time cuts the Sharpe from 0.64 to 0.40 over the common 2009+ window (about 37% of the edge), and the remaining FF6 alpha is not significant (t = 1.36).
* **Verdict:** deep learning *improved the decision* but also *re-introduced unintended risk*, the exact thing the project set out to remove. A classical constrained optimiser remains the cleanest way to isolate intended bets in this universe.

> ✍️ **YOUR ANALYSIS (2–3 sentences):** In your own words, what would you tell a PM who asks "should we put this in production?" Consider the t-stat, the beta leak and the 2022 drawdown.

---

## 1. Hypothesis

A long-short book should earn returns from its *intended* stock views, not from accidental exposure to the market, sectors or style factors. We test two claims:

* **H1:** a neural covariance model captures time-varying co-movement better than shrinkage estimators, giving better-hedged portfolios.
* **H2:** training the forecasts *through* the optimiser, on realised risk-adjusted return net of costs, produces better portfolios than "predict-then-optimise".

## 2. Data

* **Prices:** Yahoo Finance, 2003–2026, dividend- and split-adjusted. Canadian names use their NYSE listings, so everything is in USD on one calendar.
* **Factors:** Ken French data library (FF5 + momentum, daily).
* **Universe:** 100 stocks chosen as *today's* large caps (76 US, 24 Canada). One (TECK) was dropped for a stale-price run.
* **Quality checks:**
  * 114 splits verified as back-adjusted.
  * 5 outlier days, all genuine (e.g. Morgan Stanley +87% on 13 Oct 2008), kept and not clipped.
  * Zero missing days after listing.
* ⚠️ **Survivorship bias.** No point-in-time index membership was available (no CRSP). The universe is today's winners, which flatters long-only results (equal weight shows an FF6 "alpha" of 2.9%, t = 3.8) and penalises the low-volatility signal (high-volatility survivors outperformed). Sector labels are also current, not point-in-time.

## 3. Method

* **Signals:** 12-1 momentum, 1-month reversal and 1-year low volatility, each a cross-sectional z-score with equal weights fixed in advance. Out-of-sample rank ICs:
  * momentum +0.016 (t = 1.15)
  * reversal +0.002 (t = 0.12)
  * low-vol −0.036 (t = −2.08)
  * composite −0.004
  
  The composite has no edge on its own, which makes this a clean test of *construction*.
* **Shared optimiser:** every long-short strategy solves the same convex problem:

  > max μ′w − γ w′Σw − κ‖Δw‖₁, subject to Σw = 0, |wᵢ| ≤ 5%, ‖w‖₁ ≤ 2, |sector net| ≤ 5%, |β_mkt, β_SMB, β_HML| ≤ 0.05 and turnover ≤ 100% a month.

  Only the expected returns μ and the risk model Σ change between strategies.
* **Deep risk model:**
  * Each stock's last 60 days, alongside six factor returns, are encoded by a patch transformer (weekly tokens).
  * Stocks then attend to each other through a permutation-equivariant set transformer.
  * The output is Σ = D(BB′ + Ψ)D with 8 latent factors, positive definite by construction.
  * It is trained on Gaussian negative log-likelihood and starts from a trailing-volatility prior.
* **End-to-end model:**
  * The optimiser is embedded as a cvxpylayers layer.
  * An alpha network adds a *zero-initialised residual* to the classical expected return (IC × volatility × score). On day one it *is* the classical strategy.
  * The risk network is fine-tuned jointly.
  * Loss = −(realised next-month return − costs) + λ · realised daily variance.
* **Walk-forward:** yearly refits on an expanding window from Jan 2008, with a label-overlap embargo and asserted label ≤ refit date.
* **Look-ahead checks:**
  * Truncation invariance on real data, for both the backtest engine and the deep features.
  * A "cheating oracle" that trades on next month's returns scores a Sharpe of 10.6, versus 0.58 here.

## 4. Results (Jan 2008 – Sep 2026, net of 10 bps)

| Strategy | Sharpe | Return | Vol | Max DD | FF6 α (t) | Risk from stock bets |
|---|---|---|---|---|---|---|
| Equal weight (long-only) | 0.72 | 12.8% | 19.1% | −48% | 2.9% (3.8)* | 1% |
| HRP (long-only) | 0.72 | 10.9% | 16.1% | −41% | 1.8% (2.1)* | 2% |
| L/S naive signal | −0.40 | −8.7% | 18.4% | −88% | −7.8% (−3.5) | 17% |
| **L/S classical (LW + optimiser)** | −0.03 | −0.6% | 8.0% | −32% | −1.2% (−0.7) | **61%** |
| L/S deep risk + optimiser | −0.05 | −0.9% | 9.5% | −36% | −2.0% (−1.2) | 41% |
| **L/S end-to-end deep** | **0.58** | **4.7%** | 8.6% | −28% | 3.1% (1.5) | 47% |

\*Inflated by survivorship bias. "Risk from stock bets" = ex-ante stock-specific share of variance under an *independent* Barra-style evaluation model (6 style + 11 sector factors).

**Headline: risk decomposition.** The optimiser, not the neural network, is what converts a factor-driven book into a stock-picking book. Both deep variants let more *style* risk through (56% and 48%, vs 34% for classical). The likely reason is that an 8-factor covariance under-represents small but persistent co-movements that Ledoit-Wolf's full-rank matrix hedges away.

**Stress years:**

| Strategy | 2008 | 2020 | 2022 |
|---|---|---|---|
| End-to-end | +0.6% | +9.1% | **−11.3%** |
| Classical | +4.7% | +6.1% | −2.9% |

The 2022 loss is consistent with the learned tilt toward high-volatility, low-profitability stocks (the excess over classical loads −0.41 on RMW) during the rate shock.

**Robustness:**
* **Seeds.** The end-to-end Sharpe is 0.58 / 0.46 / 0.50 for seeds 42 / 43 / 44: mean 0.51 ± 0.06, with FF6 alpha t between 0.9 and 1.5. The headline seed is the best of the three. Every seed beats classical by a wide margin, and none is significant on alpha.
* **Model size.** The deep risk model with 8K / 29K / 109K parameters (d_model 16 / 32 / 64) has out-of-sample NLL of −2.919 / −2.941 / −2.931, all better than Ledoit-Wolf (−2.899). Min-variance volatility is 14.3% / 14.0% / 14.3%, all worse than Ledoit-Wolf (13.3%). The finding does not depend on capacity.
* **Constraints.** The end-to-end Sharpe stays between 0.54 and 0.67 across position caps of 3–10% and sector limits of 2.5–10%, using the trained forecasts.
* **Costs.** It falls to 0.45 at 20 bps and 0.19 at 40 bps (same trades re-priced).
* **Reproducibility.** Retraining the full walk-forward deep risk model from scratch reproduces the main run's numbers exactly (all printed digits).

> ✍️ **YOUR ANALYSIS:** Look at `reports/figures/stage6_factor_exposures.png`. Why might an optimiser trained on 2009–2021 data learn to sit at the *upper* edge of the market-beta band? Is that "learning" or "overfitting to a bull market"?

## 5. Limitations

* **Survivorship and look-back selection bias in the universe.** This is the largest caveat. Fix it with CRSP point-in-time membership.
* **Returns-based exposures.** Size and value exposures are rolling betas, not fundamental characteristics. Beta estimation error is exactly how the end-to-end model leaked market risk.
* **Small training sets for the end-to-end model.** 36 to 252 monthly decisions per refit. Effective sample sizes are small, so the improvement is not statistically established.
* **Sensitivity is incomplete.** The cost and constraint sensitivity re-uses fixed forecasts rather than retraining. Model-size sensitivity retrains only the risk model.
* **Execution assumptions.** Trades fill at the monthly close at a flat 10 bps. There's no borrow cost on shorts, no market impact and no capacity analysis.

## 6. Next steps

1. **Close the beta leak:**
   * shrink betas (Vasicek) or take them from the learned risk model
   * add a realised-beta penalty to the end-to-end loss
   * or neutralise on a forward-looking beta
2. **Use point-in-time data:** universe, sectors, and fundamental size and value.
3. **Fix the loss scale:** add a turnover-aware Sharpe loss and multiple-seed ensembling, then re-test significance with a Deflated Sharpe Ratio given the number of variants tried.
4. **Try a richer risk model:** more latent factors, or a hybrid "Ledoit-Wolf + learned correction" that keeps LW's hedging precision.

> ✍️ **YOUR ANALYSIS:** Which *one* of these would you do first at a pension fund, and why? (Think about the fund's horizon, its leverage constraints and how a CIO would read a t-stat of 1.7.)
