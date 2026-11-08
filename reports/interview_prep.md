# Interview prep: Deep Portfolio Constructor, stage by stage

For each stage: what was built, the finance in plain English, and three interview questions with model answers. Numbers refer to the results in `stage6_evaluation.md`.

---

## Stage 0: Setup and reproducibility

**What and why.**
* **One strict settings file.** Every setting lives in `config.yaml`, and the loader rejects unknown or missing keys, so a typo can't silently become a default.
* **One master seed.** It drives numpy, torch and Python's `random`.
* **Deterministic torch.** The kernels are deterministic and the thread count is fixed.
* **Proof it works.** Retraining the full walk-forward deep model from scratch reproduced every printed digit of the main run.
* **Gradient check.** I verified cvxpylayers' gradients against a problem with a known answer (dollar-neutral projection, whose Jacobian is I − 11ᵀ/n) before trusting it in Stage 5.

**Q1. Why fix seeds? Doesn't that hide instability?**
Seeds give auditability: the same code and data give the same number. To *measure* instability, I reran the end-to-end model with three seeds: Sharpe 0.58 / 0.46 / 0.50. The honest number to quote is the mean, 0.51 ± 0.06. The headline seed happened to be the best of the three.

**Q2. Why put every parameter in a config file?**
* It can be audited: all assumptions are in one place.
* Sensitivity tests become config edits, not code changes.
* It limits silent tuning: parameter choices are visible and versioned in git.

**Q3. How do you know the differentiable optimiser's gradients are right?**
* **A known answer.** I checked it on a problem with a closed-form Jacobian.
* **The real problem.** I compared autograd against central finite differences on the full constrained problem (sector, factor, caps, turnover), which is a test in `tests/test_deep.py`.

---

## Stage 1: Data

**What and why.**
* **Sources.** 100 large caps (76 US, 24 Canadian) from Yahoo Finance, plus Ken French factors.
* **Currency.** Canadian names use their NYSE listings, so everything is in USD on one trading calendar and lines up with US factors.
* **Quality checks:**
  * 114 splits verified as back-adjusted.
  * 5 outliers flagged, all real events (e.g. Morgan Stanley +87%, Oct 2008), kept rather than clipped.
  * TECK dropped for a stale-price run.
* **Point-in-time eligibility.** A stock becomes investable only after a year of clean data.

**Q1. What is survivorship bias, and how does it show up in *your* results?**
Picking today's large caps means only including the winners. It shows up in two places:
* **Long-only returns look too good.** Equal weight shows an FF6 "alpha" of 2.9% with t = 3.8, which is mostly survivorship.
* **Low volatility looks bad.** High-volatility survivors like NVDA and SHOP did very well, so the low-vol signal has a negative IC.

The fix is point-in-time index membership from CRSP.

**Q2. How did you handle stock splits?**
Yahoo's `Close` column is already split-adjusted, and `Adj Close` also adjusts for dividends. On every split date, I check that both series are smooth. A failed adjustment would show a move of about 1/ratio − 1 (−50% for a 2:1 split). My first version of the check wrongly expected `Close` to be unadjusted. The test failures showed that, and that's why I don't trust a check until I've watched it fail on purpose.

**Q3. Why use US listings for Canadian companies?**
* **One currency.** Otherwise every Canadian bet would carry CAD/USD currency risk.
* **One calendar.** US and Canadian market holidays differ.
* **Consistent factors.** It's consistent with the US Fama-French factors.

The cost is a slightly different liquidity profile from the Toronto listing.

---

## Stage 2: Signals

**What and why.**
* **The signals:** 12-1 momentum, 1-month reversal and 1-year low volatility. Each is a cross-sectional z-score, winsorised iteratively, then combined with equal weights fixed in advance.
* **How I measured them:** monthly rank IC with Newey-West t-statistics.
* **Out-of-sample ICs:**
  * momentum +0.016 (t = 1.15)
  * reversal +0.002
  * low-vol −0.036 (t = −2.08)
  * composite about 0

**Q1. What is an information coefficient, and is 0.02 good?**
The IC is the rank correlation between this month's signal and next month's return. By Grinold's fundamental law, IR ≈ IC × √breadth. With about 95 stocks rebalanced monthly, an IC of 0.02 is respectable *if it's stable*. Here the t-statistic is only 1.15, so it isn't.

**Q2. Your composite had zero IC. Why not re-weight the signals?**
Choosing weights after seeing the full-sample ICs is look-ahead bias. The weights were fixed in advance. Learning the weights walk-forward is exactly what the end-to-end model does: it learned to down-weight low volatility using only past data.

**Q3. Why Newey-West standard errors?**
Monthly IC and return series are autocorrelated (persistent signals, overlapping information), and the plain t-statistic overstates significance when that happens. HAC errors correct for it.

---

## Stage 3: Classical baselines and the engine

**What and why.**
* **One engine.** A single walk-forward engine handles monthly rebalancing, daily drift, 10 bps per unit traded and turnover tracking.
* **Long-only baselines:** equal weight, min variance, mean-variance with Ledoit-Wolf, risk parity (Spinu's convex form) and HRP.
* **Long-short:** a naive signal portfolio, and a classical version that uses the same signal in a constrained optimiser.

**Q1. Why does Ledoit-Wolf shrinkage help?**
With 99 stocks and 252 days, the sample covariance has about 5,000 parameters and noisy small eigenvalues. An optimiser that inverts it piles into combinations that only *look* riskless. Shrinking toward a structured target trades a little bias for a lot less variance. The shrinkage intensity is chosen analytically.

**Q2. What does HRP do differently?**
It never inverts the covariance matrix. It works in three steps:
1. Cluster the stocks by correlation distance.
2. Order them so similar stocks sit next to each other.
3. Split the list in halves recursively, giving each half weight in inverse proportion to its risk.

That makes it robust to estimation noise. On a diagonal covariance it reduces exactly to inverse-variance weights, which is one of the tests.

**Q3. How do you prevent look-ahead bias in a backtest?**
* **Point-in-time access.** Strategies only see a `PointInTimeView` that slices data at the decision date.
* **Truncation invariance.** Scrambling every return after a date T must leave every decision up to T unchanged. This is tested and also checked at runtime on the real data.
* **Oracle calibration.** A strategy that cheats by trading on next month's returns scores a Sharpe of 10.6, which shows what a leak looks like.

---

## Stage 4: Deep risk model

**What and why.**
* **Inputs.** Each stock's last 60 days of returns, plus six factor returns, grouped into weekly patches.
* **Temporal encoder.** A patch transformer turns each stock's history into an embedding.
* **Cross-sectional attention.** Self-attention *across stocks* lets the model learn co-movement. It is permutation-equivariant: relabelling the stocks just relabels the output, which is tested.
* **Output.** Σ = D(BB′ + Ψ)D with 8 latent factors, positive definite by construction.
* **Training.** Gaussian negative log-likelihood, yearly walk-forward refits, and an embargo between training and validation.

**Results:**
* Best out-of-sample likelihood: −2.941 vs Ledoit-Wolf's −2.899.
* But a worse min-variance portfolio: 14.0% vs 13.3% volatility.
* This held across 8K–109K parameters.

**Q1. Why the factor parameterisation?**
* **Always valid.** It is positive definite by construction, with no eigenvalue clipping.
* **Cheap.** Log-determinant and inverse take O(NK²) time via the Woodbury identity.
* **Interpretable.** B holds learned risk factors and Ψ is stock-specific risk.
* **Sensible starting point.** At initialisation it equals the trailing-volatility diagonal, so the network has to *earn* any correlation it adds.

**Q2. Your model wins on likelihood but loses on min-variance volatility. How?**
Likelihood mostly rewards getting the *level* of each stock's variance right through time (volatility timing). Min-variance portfolios are mostly sensitive to the *small eigenvalues*, the precise hedging relationships. An 8-factor model compresses those away, while Ledoit-Wolf's full-rank matrix keeps them. Better forecast, worse decision: that's the motivation for Stage 5.

**Q3. How do you avoid leakage when training on overlapping 21-day targets?**
* **Labels end by the refit date.** A sample is usable only if its full label window ended by the refit date. This is asserted in code.
* **Embargo.** The last training origin must be at least 21 days before the first validation origin, so labels don't overlap across the split.

---

## Stage 5: End-to-end differentiable optimiser

**What and why.**
* **Optimiser layer.** The same constrained problem every long-short strategy uses, wrapped in cvxpylayers. Gradients come from the KKT optimality conditions.
* **Alpha network.** It adds a *zero-initialised* residual to the classical expected return (IC × volatility × score). At step 0 the model *is* the classical strategy.
* **Risk network.** Starts from the Stage 4 checkpoint and is fine-tuned with a smaller learning rate.
* **Loss.** −(realised net return − costs) + λ × realised daily variance.

**Results:**
* Net Sharpe 0.58 (seed mean 0.51) vs −0.03 for classical.
* Edge t = 1.70.
* Realised beta 0.19.

**Q1. Why train end-to-end instead of predict-then-optimise?**
Forecast error is not decision error. The optimiser is very sensitive to some errors (e.g. small eigenvalues, the ranking of the extreme stocks) and ignores others. Training on realised portfolio utility puts the model's capacity where the decision is sensitive. The risk is that it also finds *unintended* routes to utility, which is exactly what happened with beta.

**Q2. What did the model learn?**
* **Composite scale:** cut from 1.0 to 0.76.
* **Signal weights:** reduced momentum (−0.07), reversal (−0.07) and especially low volatility (−0.14), in line with low-vol's negative IC.
* **Market beta:** pressed against the *upper* edge of its ±0.05 beta band. In a 2009–2021 bull market, being as long beta as allowed paid off. Beta estimation error in high-volatility names then let realised beta reach 0.19.

**Q3. How did you keep it from overfitting with so few monthly samples?**
* **Zero-initialised residual,** so it starts at the classical answer.
* **Early stopping** on a 12-month validation window.
* **Warm starts** from the previous year.
* **Weight decay,** plus a smaller learning rate for the pretrained risk network.

Even so, the improvement is not significant (t = 1.7), which I report.

---

## Stage 6: Evaluation

**What and why.**
* **Standard metrics:** Sharpe, Sortino, max drawdown, hit rate, turnover and FF6 alpha with HAC t-statistics.
* **Headline risk decomposition:** an *independent* Barra-style model (6 style betas and 11 sectors, with cross-sectional regressions each day) splits risk into stock-specific and factor parts. The share from intended stock bets:
  * naive 17%
  * classical 61%
  * deep two-stage 41%
  * end-to-end 47%
* **Stress years:** 2008, 2020 and 2022 shown separately.
* **Sensitivity:** costs, constraints, model size and seeds.
* **Beta-leakage diagnostic.** Hedging each strategy's own trailing beta, point-in-time, takes the end-to-end Sharpe from 0.64 to 0.40 on the common window.

**Q1. Why use a separate risk model for the decomposition?**
Using the deep model to grade itself is circular: a model that underestimates its own factor risk would look great. An independent model built from observable exposures gives a neutral referee.

**Q2. Is the end-to-end result real?**
* **Not established.** The edge t is 1.70. The beta-hedged alpha has t = 1.36. Across seeds the Sharpe is 0.51 ± 0.06.
* **Partly beta.** About 37% of the edge is leaked market beta.
* **Style tilt.** The extra return loads −0.41 on profitability (RMW), a tilt toward less-profitable stocks, and it lost 11% in 2022.
* **Multiple testing.** Several variants were tried during development, so a deflated Sharpe ratio would cut it further.

My honest answer: promising, not proven.

**Q3. The constraint says beta ≤ 0.05, yet realised beta is 0.19. How?**
The constraint uses *estimated* betas from a trailing 252-day regression. For volatile stocks those estimates are noisy and shrink toward the past. An optimiser rewarded for return will exploit that estimation error, systematically favouring stocks whose true beta is above their estimate. Possible fixes:
* Vasicek-shrunk betas
* betas taken from the learned risk model
* a realised-beta penalty in the loss

---

## Stage 7: Dashboard and report

**Q1. What would you show a PM first?**
The risk-breakdown tab. It answers "what am I being paid for?" before "how much did I make?" For a pension fund, an unintended beta of 0.19 on a "market-neutral" sleeve matters more than an extra 0.1 of Sharpe.

**Q2. Would you put the end-to-end model in production?**
Not yet. I'd run it in shadow alongside the classical book with three changes: shrunk betas, a realised-beta penalty, and point-in-time data. Then I'd decide on deflated-Sharpe evidence.

**Q3. What did this project teach you?**
> ✍️ *Your answer, in your own words. Interviewers can tell.*
