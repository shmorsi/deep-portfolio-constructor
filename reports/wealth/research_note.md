# The Deep Growth Network: a pre-registered test of deep learning for long-term, long-only investing

Sherif Morsi · Deep Portfolio Constructor, commit 9eab488 · every number below is generated from the committed results by `scripts/write_research_note.py`.

**Abstract.** A transformer allocates a long-only, fully invested portfolio across Kenneth French's 30 US industry portfolios and learns what fraction of the gap to its target to trade each month. It is trained in PyTorch on minus the log growth of wealth after trading costs, through a differentiable simulator of the investing rules, walk-forward with refits every five years. Out of sample from January 1960 to September 2026 (801 months, 10 bps per unit traded), $1 grew to $1,368 with the network and $929 in the market: +0.58% a year (Newey-West t 1.40, 95% interval −0.07% to +1.50%): no detectable difference. In words: the network's edge over holding the market is too small to tell from luck by the pre-registered test. The test was fixed in git before the network was run out of sample.

## 1. Question

Can a neural network, trained on the growth of wealth after trading costs, build a better long-only, fully invested, long-term portfolio than simply holding the market, and than the classical rules? The protocol ([`docs/wealth_protocol.md`](../../docs/wealth_protocol.md)) fixed the data, the rules, the model, the comparisons and the decision rule before the out-of-sample run. The primary metric is the annualised difference in net log growth, network minus market; "better" needs a Newey-West t of at least 2 (12 lags) and a 95% moving-block bootstrap interval (12-month blocks, 5,000 draws) that excludes zero.

## 2. Data and rules

Kenneth French's 30 value-weighted industry portfolios, July 1926 to September 2026. Each holds every listed firm in its industry at each date, so there is no survivorship bias among firms. Industry market caps give the market portfolio, which tracks French's own market return with a correlation of 0.9998. Every strategy is long-only and fully invested, with no leverage and no cash; a decision at the end of month t is held through month t+1, holdings drift with prices between trades, and every unit traded costs 10 bps.

## 3. Model and training

Each industry is a token of 13 point-in-time features (returns over 1, 3, 12-1, 36 and 60 months, volatility, downside risk, drawdown, beta and correlation with the market, book-to-market, index weight) plus its current holding relative to the market; a market token carries 7 market-wide features. Two transformer blocks (d = 32, 4 heads; 20,066 parameters) let every industry attend to every other. A tilt head sets the target `softmax(log w_mkt + log 3 · tanh(score))`, so each market weight is scaled by a factor between a third and three before rescaling; it starts at zero, so the untrained network is the index. A trade-fraction head sets κ, and new holdings are `w + κ (target − w)`.

Training plays the policy through a differentiable copy of the backtest engine (tested to reproduce it month by month) over 60-month episodes and minimises minus the average monthly log growth after costs, the Kelly criterion. Refits every five years from December 1959 to December 2024 use only earlier data; the last ten years of each window are validation for early stopping, and no training label is dated after the first validation decision. Five seeds are warm-started from their own previous refits, and the deployed policy averages their targets and trade fractions.

## 4. Results

| Strategy | $1 became | CAGR | Volatility | Sharpe | Max drawdown | Traded a year | Against the market |
|---|---:|---:|---:|---:|---:|---:|---|
| Market (hold the index) | $929 | 10.78% | 15.3% | 0.47 | −50% | 4% | benchmark |
| Equal weight | $1,377 | 11.44% | 16.3% | 0.49 | −53% | 6% | +0.59% a year (Newey-West t 0.87, 95% interval −0.65% to +2.04%): no detectable difference |
| Minimum variance | $1,180 | 11.18% | 12.9% | 0.56 | −41% | 20% | +0.36% a year (Newey-West t 0.33, 95% interval −1.73% to +2.35%): no detectable difference |
| Risk parity | $1,418 | 11.49% | 15.3% | 0.51 | −49% | 7% | +0.63% a year (Newey-West t 0.94, 95% interval −0.59% to +2.04%): no detectable difference |
| Classical growth optimiser | $905 | 10.74% | 15.3% | 0.47 | −48% | 9% | −0.04% a year (Newey-West t −0.09, 95% interval −0.81% to +0.85%): no detectable difference |
| Momentum tilt | $1,764 | 11.85% | 15.6% | 0.53 | −46% | 110% | +0.96% a year (Newey-West t 2.93, 95% interval +0.28% to +1.56%): better |
| Deep Growth Network | $1,368 | 11.43% | 15.1% | 0.51 | −52% | 155% | +0.58% a year (Newey-West t 1.40, 95% interval −0.07% to +1.50%): no detectable difference |

**Primary test.** Network minus market: +0.58% a year (Newey-West t 1.40, 95% interval −0.07% to +1.50%): no detectable difference. The secondary comparisons, reported but not used to choose anything: against equal weight, −0.01% a year (Newey-West t −0.01, 95% interval −1.21% to +1.23%): no detectable difference; against the momentum tilt, −0.38% a year (Newey-West t −0.69, 95% interval −1.28% to +0.85%): no detectable difference; against the classical growth optimiser, +0.62% a year (Newey-West t 1.05, 95% interval −0.46% to +1.88%): no detectable difference.

**Over time.** The network grew faster than the market in 79% of rolling ten-year windows (the momentum tilt in 99%). Its Fama-French three-factor alpha was +0.15% a year (t 0.38), with a market beta of 0.98. Its worst calendar year was −39%, against −36% for the market.

**Costs and seeds.** Re-pricing the same decisions, its growth against the market (also re-priced) was +0.88% a year with free trading, +0.13% at 25 bps and −0.63% at 50 bps. Each seed's network on its own turned $1 into $1,794, $574, $1,731, $1,091, $1,066.

## 5. What the network learned

Averaged over the 801 months, its largest overweights were steel (1.48x), oil and gas (1.32x), mining (1.31x), and its largest underweights consumer goods (0.60x), beer and liquor (0.62x), business services (0.64x) (its average weight over the market's average weight). Its monthly active return (network minus market, in log growth) had a correlation of 0.07 with the momentum tilt's. Its trade fraction, which training starts at 25%, had a median of 81% out of sample. In the 88 months when the market stood 20% or more below its peak it averaged 81%, against 86% in other months. It traded 155% of the portfolio a year, an average holding period of 0.6 years, against 4% for the index and 110% for the momentum tilt. Its weights ranged from 0.19 to 6.37 times the market's. Its loadings on the Fama-French factors were 0.98 on the market, 0.01 on size and 0.17 on value (the market's own: −0.02 and 0.01), with a three-factor alpha of +0.15% a year (t 0.38).

## 6. Ablations

Fixed in the protocol, two seeds each (42 and 43), compared here with the full network averaged over the same two seeds ($1,026):

| Ablation | $1 became | Traded a year | Against the full network, same seeds |
|---|---:|---:|---|
| Deep, no learned holding | $1,775 | 252% | +0.82% a year (Newey-West t 1.83, 95% interval −0.11% to +1.55%): no detectable difference |
| Deep, no attention | $701 | 91% | −0.57% a year (Newey-West t −1.57, 95% interval −1.24% to +0.15%): no detectable difference |
| Deep, no market anchor | $387 | 354% | −1.46% a year (Newey-West t −0.55, 95% interval −6.69% to +3.79%): no detectable difference |

## 7. Limits

The 30 portfolios are industries, not funds an investor could have bought in 1960; costs are a flat 10 bps with no taxes or market impact. The features carry hindsight: momentum, value and low volatility were documented decades after 1960, and the pre-registration fixes the test, not the feature list. The data are one country's century, among the best-performing markets in the world, and several strategies were compared; the pre-registered primary test is the one to read.

## 8. Conclusion

Trained on the right objective, with a market anchor and a learned trade fraction, the network did not separate from holding the market by a test fixed in advance. For a long-only investor in 30 broad industries, the index is a hard benchmark; the learned tilts neither clearly helped nor clearly hurt. Two things it learned are worth knowing. Given the choice, it did not sit still: its trade fraction rose from 25% to a median of 81%, so the 10 bps cost per trade was not enough to teach patience on these features. And its gain over the market is about what its value tilt would have paid: a value loading of 0.17 and a three-factor alpha of +0.15% (t 0.38). A fresh test needs fresh data: the honest next step is forward, publishing the network's holdings each month from now on and scoring them as the months arrive, under the same protocol.
