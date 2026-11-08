# Deep Portfolio Construction: Can Deep Learning Keep the Bets We Want and Drop the Ones We Don't?

**Author:** Sherif Morsi · **Date:** November 2026 · **Universe:** 99 US & Canadian large caps · **Sample:** Jan 2008 – Sep 2026, out-of-sample, monthly, net of 10 bps

---

## Bottom line

* **Portfolio construction matters more than the model.** The same three signals traded naively lose 8.7% a year, with 83% of their risk coming from factor and sector exposure nobody intended. Put through a constrained optimiser (dollar-, sector-, beta-, size- and value-neutral), the share of risk from intended stock bets rises from **17% to 61%** and volatility falls from 18% to 8%.
* **A five-step ablation measured each deep component (all PyTorch):**
  * deep risk model: −0.05
  * deep views (AlphaFormer): 0.33
  * end-to-end training through the optimiser: **0.58**
  * fully deep end-to-end: 0.15
  
  That compares with −0.03 for the classical optimiser. End-to-end training helped most, and stacking the deep return model on top of it hurt (t = −1.90 against the end-to-end model).
* **The best result is suggestive, not conclusive, and partly leaked beta.** The end-to-end edge over classical is +0.43% a month (Newey-West t = 1.70). The model learned to press against its +0.05 beta limit, and realised beta was 0.19. Hedging its own trailing beta point-in-time cuts the Sharpe from 0.64 to 0.40 over the same window.
* **Verdict:** deep learning *improved the decision* but also *re-introduced unintended risk*, the exact thing the project set out to remove. More deep learning was not better on 225 months of data. A classical constrained optimiser remains the cleanest way to isolate intended bets in this universe.

> ✍️ **YOUR ANALYSIS (2–3 sentences):** In your own words, what would you tell a PM who asks "should we put this in production?" Consider the t-stat, the beta leak, the fully deep model's overfitting and the 2022 drawdown.

---

## 1. Hypothesis

A long-short book should earn returns from its *intended* stock views, not from accidental exposure to the market, sectors or style factors. We test three claims:

* **H1:** a neural covariance model captures time-varying co-movement better than shrinkage estimators, giving better-hedged portfolios.
* **H2:** a neural return model extracts more from the same inputs than a fixed signal composite.
* **H3:** training forecasts *through* the optimiser, on realised risk-adjusted return net of costs, produces better portfolios than "predict-then-optimise".

## 2. Data

* **Prices:** Yahoo Finance, 2003–2026, dividend- and split-adjusted. Canadian names use their NYSE listings, so everything is in USD on one calendar.
* **Factors:** Ken French data library (FF5 + momentum, daily).
* **Universe:** 100 stocks chosen as *today's* large caps (76 US, 24 Canada). One (TECK) was dropped for a stale-price run.
* **Quality checks:**
  * 114 splits verified as back-adjusted.
  * 5 outlier days, all genuine, kept and not clipped.
  * Zero missing days after listing.
* ⚠️ **Survivorship bias.** No point-in-time index membership was available (no CRSP). The universe is today's winners, which flatters long-only results (equal weight shows an FF6 "alpha" of 2.9%, t = 3.8) and penalises the low-volatility signal. Sector labels are also current, not point-in-time.

## 3. Method

* **Signals:** 12-1 momentum, 1-month reversal and 1-year low volatility, each a cross-sectional z-score with equal weights fixed in advance. The composite's out-of-sample rank IC is −0.004, so it has no edge on its own. That makes this a clean test of *construction*.
* **Shared optimiser:** every long-short strategy solves the same convex problem:

  > max μ′w − γ w′Σw − κ‖Δw‖₁, subject to Σw = 0, |wᵢ| ≤ 5%, ‖w‖₁ ≤ 2, |sector net| ≤ 5%, |β_mkt, β_SMB, β_HML| ≤ 0.05 and turnover ≤ 100% a month.

  Only the expected returns μ and the risk model Σ change between strategies.
* **AlphaFormer (views, PyTorch):**
  * A transformer reads each stock's last 12 monthly returns, then self-attention runs *across stocks* (permutation-equivariant).
  * It is trained on the negative information coefficient.
  * It is a 3-seed ensemble with 28,033 parameters per member.
* **Deep risk model (PyTorch):**
  * A patch transformer reads each stock's last 60 days alongside six factor returns, then cross-stock attention.
  * The output is Σ = D(BB′ + Ψ)D, positive definite by construction.
  * It is trained on Gaussian negative log-likelihood and has 29,033 parameters.
* **End-to-end models:**
  * The optimiser is embedded as a cvxpylayers layer.
  * An alpha network adds a *zero-initialised residual* to the expected return, so on day one it *is* the two-stage strategy.
  * The pretrained networks fine-tune jointly.
  * Loss = −(realised next-month return − costs) + λ · realised daily variance.
* **Walk-forward:** yearly refits on an expanding window from Jan 2008, with label-date assertions and embargoes for every model.
* **Look-ahead checks:**
  * Truncation invariance on real data, for both the backtest engine and the deep-model inputs.
  * A cheating oracle scores a Sharpe of 10.6, versus 0.58 here.

## 4. Results (Jan 2008 – Sep 2026, net of 10 bps)

| Strategy | Sharpe | Return | Vol | Max DD | FF6 α t | Beta | Risk from stock bets |
|---|---|---|---|---|---|---|---|
| Equal weight (long-only) | 0.72 | 12.8% | 19.1% | −48% | 3.83* | 0.92 | 1% |
| L/S naive signal | −0.40 | −8.7% | 18.4% | −88% | −3.47 | −0.16 | 17% |
| **L/S classical (LW + optimiser)** | −0.03 | −0.6% | 8.0% | −32% | −0.69 | 0.02 | **61%** |
| + deep risk model | −0.05 | −0.9% | 9.5% | −36% | −1.24 | 0.05 | 41% |
| + deep views (AlphaFormer) | 0.33 | 2.5% | 8.5% | −26% | 0.80 | 0.15 | 50% |
| **End-to-end through the optimiser** | **0.58** | **4.7%** | 8.6% | −28% | 1.48 | 0.19 | 47% |
| Fully deep end-to-end | 0.15 | 1.0% | 9.3% | −31% | 0.04 | 0.17 | 48% |

\*Inflated by survivorship bias. "Risk from stock bets" = ex-ante stock-specific share of variance under an *independent* Barra-style evaluation model (6 style + 11 sector factors).

**Headline: risk decomposition.** The optimiser, not the neural network, is what converts a factor-driven book into a stock-picking book. Every deep variant lets more *style* risk through than the classical optimiser (41–50% stock-specific, vs 61%).

**Each hypothesis, in turn:**
* **H1, rejected for hedging.** The deep covariance model wins on likelihood (−2.941 vs −2.899 for Ledoit-Wolf) but builds a *more* volatile minimum-variance portfolio (14.0% vs 13.3%), at every model size tried.
* **H2, weakly supported.** AlphaFormer's out-of-sample IC is 0.0165 (t = 1.25), against −0.004 for the composite. Its top-minus-bottom quintile spread is +0.47% a month (t = 1.63).
* **H3, supported but not significant.** End-to-end training lifted Sharpe to 0.58 (t = 1.70 against classical). Training AlphaFormer end to end as well overfit: about 250 monthly decisions per refit cannot support the extra parameters.

**Stress years:**

| Strategy | 2008 | 2020 | 2022 |
|---|---|---|---|
| End-to-end | +0.6% | +9.1% | **−11.3%** |
| Classical | +4.7% | +6.1% | −2.9% |

The 2022 loss is consistent with the learned tilt toward high-volatility, low-profitability stocks (the end-to-end model's excess over classical loads −0.41 on RMW) during the rate shock.

**Robustness:**
* End-to-end Sharpe across three seeds: 0.58 / 0.46 / 0.50, so the headline seed is the best of the three.
* Across position caps of 3–10% and sector limits of 2.5–10%: 0.54–0.67.
* At 20 and 40 bps costs: 0.45 and 0.19.

> ✍️ **YOUR ANALYSIS:** Look at `reports/figures/stage6_factor_exposures.png`. Why might an optimiser trained on 2009–2021 data learn to sit at the *upper* edge of the market-beta band? Is that "learning" or "overfitting to a bull market"?

## 5. Limitations

* **Survivorship and look-back selection bias in the universe.** This is the largest caveat. Fix it with CRSP point-in-time membership.
* **Returns-based exposures.** Size and value exposures are rolling betas, not fundamental characteristics. Beta estimation error is exactly how the end-to-end models leaked market risk.
* **Small training sets and many variants.** 36 to 252 monthly decisions per refit. Six market-neutral variants were compared, so the best one's t-stat needs a multiple-testing haircut, and the fully deep model was run with one seed.
* **Sensitivity is incomplete.** The cost and constraint sensitivity re-uses fixed forecasts rather than retraining.
* **Execution assumptions.** Trades fill at the monthly close at a flat 10 bps. There's no borrow cost on shorts, no market impact and no capacity analysis.

## 6. Next steps

1. **Close the beta leak:**
   * shrink betas (Vasicek) or take them from the learned risk model
   * add a realised-beta penalty to the end-to-end loss
   * or neutralise on a forward-looking beta
2. **Use point-in-time data:** universe, sectors, and fundamental size and value.
3. **Regularise the fully deep model:** freeze AlphaFormer and train only its output scale end to end, pool training across more assets, or ensemble seeds. Then re-test significance with a Deflated Sharpe Ratio given the number of variants tried.
4. **Try a richer risk model:** a hybrid "Ledoit-Wolf + learned correction" that keeps LW's hedging precision.

> ✍️ **YOUR ANALYSIS:** Which *one* of these would you do first at a pension fund, and why? (Think about the fund's horizon, its leverage constraints and how a CIO would read a t-stat of 1.7.)
