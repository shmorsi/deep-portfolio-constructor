# Deep Portfolio Constructor

**A neural network that builds a long-only portfolio for long-term wealth.** A transformer allocates across Kenneth French's 30 US industry portfolios and decides how much to trade each month. It is trained in PyTorch on the growth of wealth itself, after trading costs, through a differentiable simulator of buying, holding and paying to trade, and tested out of sample from January 1960 to 2026 against the market and five classical long-only strategies, under a protocol committed to git before the test.

**Explore it:** [shmorsi.github.io/projects/deep-portfolio-constructor.html](https://shmorsi.github.io/projects/deep-portfolio-constructor.html). Slide the growth toy, pick a strategy and a start year, and scrub through the network's holdings since 1960.

**One question, three studies: can AI help someone who invests for decades, and how would you know?**

| | Study | What it asks | Where |
|---|---|---|---|
| 1 | **Wealth study** | Can a network trained on the growth of wealth build a better long-only portfolio than holding the market? | this README, from the top |
| 2 | **[Pension Lab](#pension-lab-what-it-costs-to-keep-a-pension-promise)** | What does it cost to keep a pension promise funded, and does a learned policy cost less than a glide path? | this repository, `src/dpc/pension/` |
| 3 | **[World Shock Lab](https://github.com/shmorsi/world-shock-lab)** | What does a central-bank or oil surprise do to the markets a long-term investor holds? | its own repository; [the lab](https://shmorsi.github.io/shock-lab/) |

The three share one method: fix the test before running it, score on data the model never saw, and publish the answer when it is no.

```mermaid
flowchart LR
  D["30 US industries, 1926 to 2026<br/>13 point-in-time features each<br/>+ 7 market-wide features"] --> T["Transformer<br/>industries attend to each other<br/>and to a market token"]
  T -- "tilt each market weight<br/>by 1/3x to 3x" --> G["Target portfolio<br/>long-only, fully invested"]
  T -- "trade fraction κ" --> H["New holdings<br/>w + κ (target − w)"]
  G --> H
  H --> S["Differentiable simulator<br/>prices drift the holdings,<br/>every trade costs 10 bps"]
  S --> L["Loss: minus the log growth<br/>of wealth over 60 months"]
  L -. "gradient through every month of compounding" .-> T
```

## Summary

**The question** ([pre-registered](docs/wealth_protocol.md)): can a neural network, trained on the growth of wealth after trading costs, build a better long-only, fully invested, long-term portfolio than simply holding the market, and than the classical rules, from 1960 to 2026?

**The answer:** **Not detectably, by the test fixed in advance.** $1 invested in January 1960 grew to **$1,368** with the network, against **$929** for the market: +0.58% a year in log growth, a gap the pre-registered test cannot tell from luck (Newey-West t = 1.40, 95% interval −0.07% to +1.50%).

* **The rules are a long-term investor's:** long-only, fully invested, no leverage, no shorting, no market timing. Within them the network traded 155% of the portfolio a year (an average holding period of 0.6 years, against 4% a year for the index), and its weights differed from the index's by 32% on average.
* **The strongest classical rival is simple.** A momentum tilt grew +0.96% a year (t = 2.93, better) against the market, trading 110% a year. The network against it: −0.38% a year (t = −0.69, no detectable difference).
* **Classical portfolio rules did not beat holding the market.** Minimum variance, risk parity, equal weight and a classical growth (Kelly) optimiser with the network's own goal and bounds all finished within the noise of the market by the same test.
* **Its edge over the market looks like a value tilt.** Its loading on the Fama-French value factor was 0.17; after adjusting for market, size and value its alpha was +0.15% a year (t = 0.38).
* **Robustness.** Each seed's network on its own ended above the market in 4 of 5 cases ($574 to $1,794); the deployed policy averages them. Re-priced at 25 and 50 bps a trade, the network's growth against the market was +0.13% and −0.63% a year.

## Results

$1 invested in January 1960, out of sample to September 2026 (801 months), long-only, fully invested, net of 10 bps per unit traded. "Growth vs market" is the yearly difference in log growth with its Newey-West t; the verdict follows the pre-registered rule (t ≥ 2 and a 95% block-bootstrap interval that excludes zero).

| Strategy | $1 became | CAGR | Sharpe | Max drawdown | Traded a year | Growth vs market (t) | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| Market (hold the index) | $929 | 10.78% | 0.47 | −50% | 4% | benchmark |  |
| Equal weight | $1,377 | 11.44% | 0.49 | −53% | 6% | +0.59% (0.87) | no detectable difference |
| Minimum variance | $1,180 | 11.18% | 0.56 | −41% | 20% | +0.36% (0.33) | no detectable difference |
| Risk parity | $1,418 | 11.49% | 0.51 | −49% | 7% | +0.63% (0.94) | no detectable difference |
| Classical growth optimiser | $905 | 10.74% | 0.47 | −48% | 9% | −0.04% (−0.09) | no detectable difference |
| Momentum tilt | $1,764 | 11.85% | 0.53 | −46% | 110% | +0.96% (2.93) | better |
| **Deep Growth Network** | $1,368 | 11.43% | 0.51 | −52% | 155% | +0.58% (1.40) | no detectable difference |

![Growth of $1 since 1960](reports/figures/wealth/growth.png)

**What each part of the network adds.** Each ablation was fixed in the protocol and trained with seeds 42 and 43. The full network averaged over the same two seeds turned $1 into $1,026.

| Ablation | What changes | $1 became | Traded a year | Against the market | Against the full network, same seeds |
|---|---|---:|---:|---|---|
| Deep, no learned holding | trades all the way to its target every month (κ fixed at 1) | $1,775 | 252% | +0.97% a year, t 2.11: better | +0.82% a year, t 1.83: no detectable difference |
| Deep, no attention | judges each industry on its own features and the market's (no attention across industries) | $701 | 91% | −0.42% a year, t −0.95: no detectable difference | −0.57% a year, t −1.57: no detectable difference |
| Deep, no market anchor | allocates from scratch with no tilt bounds (no market anchor; untrained, it holds equal weights) | $387 | 354% | −1.31% a year, t −0.47: no detectable difference | −1.46% a year, t −0.55: no detectable difference |

The ablations were fixed in advance to be reported, not to choose a model. “Deep, no learned holding” came out better than the market on its own (t = 2.11), but it is not distinguishable from the full network with the same seeds; with this many comparisons, one pass is weak evidence.

**What it learned.** It did not learn to sit still. Its trade fraction started training at 25% and settled at a median of 81% (range 40% to 100%): it chased its moving target, trading 155% of the portfolio a year. It leaned toward value: its loading on the Fama-French value factor was 0.17 (the market's 0.01), and after adjusting for market, size and value its alpha was +0.15% a year (t = 0.38). The tilts by industry over time:

![The network's tilts by industry](reports/figures/wealth/tilts.png)

![Rolling 10-year growth against the market](reports/figures/wealth/rolling.png)

Every table (decades, costs, seeds, factor alphas, all comparisons) is in [`reports/wealth/results.md`](reports/wealth/results.md). The write-up is [`reports/wealth/research_note.md`](reports/wealth/research_note.md).

## Against the S&P 500 and the NASDAQ

Added after the pre-registered test, as context; the protocol's benchmark stays the whole US market. Comparisons use the same rule (Newey-West t and block bootstrap).

| $1 invested | S&P 500, dividends reinvested | NASDAQ Composite, price only | Whole US market | Deep Growth Network |
|---|---:|---:|---:|---:|
| End of December 1959 | $831 | | $929 | $1,368 |
| End of February 1971 | $360 | $265 | $388 | $586 |

* **S&P 500:** the network against it, +0.75% a year (t = 1.56, no detectable difference). The whole market against it, +0.17% a year (t = 0.65, no detectable difference).
* **NASDAQ Composite:** the network against it, +1.43% a year (t = 0.84, no detectable difference). The NASDAQ series is price only (no free dividend history before 2003), so these gaps flatter everything compared with it.
* **Sources.** Yahoo Finance: `^SP500TR` (S&P 500 total return) from February 1988, `^GSPC` (price) and `^IXIC` (NASDAQ Composite); before 1988 the S&P 500's dividends come from Robert Shiller's monthly data. Where both exist (1988-02 to 2023-06, 425 months), the price-plus-dividends series tracks the official total-return index with a correlation of 0.9999 and a gap of −0.072% a year. Run `python -m dpc.cli wealth-benchmarks`.

![Against the S&P 500 and the NASDAQ](reports/figures/wealth/benchmarks.png)

## How often should it trade?

Added after the pre-registered test. **The same trained network, with nothing retrained,** run on four schedules:

* **Every month**: the tested policy. At each month-end it reads the features, sets a target and moves its trade fraction of the way there.
* **Every quarter / every year**: the network is asked only at the end of March, June, September and December, or only each December; in between, the holdings drift with prices, untouched.
* **Every day**: the network still decides at each month-end (its inputs are monthly), and then at every daily close the portfolio is traded back to the weights it chose, using Kenneth French's daily industry returns from the same CRSP vintage.

Every schedule pays 10 bps per unit traded. "Before costs" re-runs the same decisions with free trading.

| Rebalanced | $1 became | Before costs | Traded a year | Cost a year | Against the market |
|---|---:|---:|---:|---:|---|
| Every day | $1,217 | $1,630 | 218% | 0.44% | +0.41% (t = 0.97): no detectable difference |
| Every month (as tested) | $1,368 | $1,685 | 155% | 0.31% | +0.58% (t = 1.40): no detectable difference |
| Every quarter | $1,162 | $1,267 | 64% | 0.13% | +0.34% (t = 0.80): no detectable difference |
| Every year | $902 | $930 | 22% | 0.05% | −0.04% (t = −0.12): no detectable difference |

Here **every month (as tested)** grew the most. Trading every day only keeps the portfolio at the month's chosen mix, and it costs more; trading less often lets the network's tilts go stale, suggesting that what it knows about each industry fades within months. A check of the daily engine: trading only at month-ends on daily data gives $1,388, against $1,368 from the monthly engine (+0.021% a year). Run `python -m dpc.cli wealth-schedules`.

![The same network on four schedules](reports/figures/wealth/schedules.png)

## Pension Lab: what it costs to keep a pension promise

<!-- pension:start -->
<!-- Generated by `python -m dpc.cli pension-site` from reports/pension/summary.json. Edit src/dpc/pension/site.py, not this block. -->

**The question** ([pre-registered](docs/pension_protocol.md), committed before the test ran): a pension plan is 90% funded and has promised 60 years of inflation-indexed payments. What does keeping that promise cost its sponsor over 20 years under the policies plans really use, and does a policy trained through a differentiable simulator of the plan cost less than a glide path?

**The answer:** In 10,000 simulated futures the learned policy cost the sponsor **$5.0** per $100 promised against **$9.9** for the glide path, and $30.2 against $69.1 in the worst 5% of futures: **better** by the pre-registered test (score difference −43.8, 95% interval −46.9 to −40.7). Over 34 real 20-year periods, with every network trained only on data from before its period, it cost **$6.3** against **$6.3**: **no detectable difference** (difference +0.1, interval −0.7 to +1.2). So the saving is a property of the simulated world the policy was trained in. Real history, the test it could not have learned, does not confirm it: there the learned policy is not detectably different from the glide path.

![The funded ratio over 20 years under 60/40, the glide path and the learned policy](reports/figures/pension/fans.png)

| Policy | Average cost | Worst 5% | Fully funded at year 20 | In stocks | When yields never come back: average, worst 5% | Real history: average, worst period |
|---|---:|---:|---:|---:|---:|---:|
| 60/40 | $14.3 | $86.0 | 90% | 60% | $18.4, $121.1 | $9.7, $53.9 |
| Liability-driven 40/60 | $10.1 | $58.5 | 91% | 40% | $14.2, $95.9 | $6.8, $42.1 |
| Fully matched | $18.2 | $62.4 | 57% | 0% | $20.6, $78.0 | $8.4, $50.6 |
| Glide path | $9.9 | $69.1 | 93% | 31% | $14.1, $113.5 | $6.3, $46.9 |
| **Learned policy** | $5.0 | $30.2 | 98% | 18% | $7.0, $49.7 | $6.3, $55.6 |

Cost is everything the sponsor pays in over 20 years plus any shortfall left at the end, per $100 promised, in start-of-plan money. Lower is better.

* **The promise is a real liability, not a return target.** 60 years of monthly payments, fully indexed to consumer prices, valued off the 10-year government yield with expected inflation (duration 14.2 years at 4%). When yields fall the promise costs more; when prices rise faster than expected it costs more again. A shortfall is paid off by the sponsor over ten years; a surplus is never paid back.
* **Three assets, no leverage.** Stocks, 10-year government bonds, and bonds matched to the promise (they move with it when yields move). Long-only, fully invested, 10 bps per dollar traded.
* **Four standard policies.** 60/40; liability-driven 40/60; fully matched; and a glide path (70% stocks while the plan is short, sliding to 20% as it fills). Fully matched is the dearest on average ($18.2): it locks the gap in and the sponsor pays it off.
* **The learned policy** is a 1,507-parameter network that starts as the glide path and is trained in PyTorch by differentiating through the plan simulator, on one score: the average cost plus the average cost of the worst 5% of futures. It sees only what a plan would know that month.
* **When yields never come back.** In the simulated futures yields are pulled toward their long-run average, and a network can learn to bet on that. Run again, without retraining, on futures where yields only drift: $7.0 against $14.1 for the glide path, **better**.
* **Real history is the check on the model.** A plan starts every December from 1972 to 2005 and lives through what happened next. The periods overlap, so they amount to about three independent ones: enough to see a policy fail, not enough to prove a small edge.
* **What it learned** (described after the test, not pre-registered). Less stock-market risk than the glide path at every funded ratio (41% in stocks under 80% funded where the glide path holds 70%; 16% over 110% funded where it holds 20%), and above all a response to the yield: for plans 85% to 100% funded it keeps 54% in stocks when the yield is under 3% and 5% when it is over 10%, the rest almost all in matching bonds. Its target weights move 69% of the fund a year.
* **Is it the network, or a better-tuned rule?** The glide path's four numbers were then trained on the same simulations for the same score. Tuned, it costs $10.9 on average and $53.5 in the worst 5%: 33% of the network's improvement in the score. Against the tuned rule the network is still **better than** it in simulation (score difference −29.2, interval −31.0 to −27.5).
* **Where the saving came from.** The simulated futures replay history in runs averaging 12 months, and the network sees the last 12 months. On futures whose months are drawn one at a time, so that there is no run to read, it costs $10.2 on average against $10.4 for the glide path and $11.7 for the tuned rule ($59.0, $79.0 and $59.9 in the worst 5%): its lead over the tuned rule falls from −29.2 to −2.4, 8% of what it was. Most of the simulated saving came from patterns in the last twelve months that hold inside the runs of history the simulator replays. A plan cannot rely on them: real history, where every network had seen only the past, gave the same answer.

![What the learned policy holds: by funded ratio at three yields, and by the yield in the test futures](reports/figures/pension/anatomy.png)

### The three studies meet: what a Fed surprise does to this plan

[World Shock Lab](https://github.com/shmorsi/world-shock-lab), the third study, measures what a surprise in US monetary policy does to markets. Its estimates (commit 9707616) are applied here to the plan's funded ratio, at 90% funded, on the announcement day, per 10 bp of surprise:

| Surprise (+10 bp) | Stocks | 10-year yield | The promise | 60/40 | Liability-driven 40/60 | Fully matched | Glide path | Learned policy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fed policy surprise | −0.98% | +3.4 bp | −0.48% | −0.20 (±0.37) | −0.18 (±0.22) | +0.00 (±0.00) | −0.24 (±0.29) | −0.13 (±0.15) |
| Fed information surprise | +0.41% | +4.1 bp | −0.57% | +0.62 (±0.79) | +0.35 (±0.46) | +0.00 (±0.00) | +0.47 (±0.61) | +0.25 (±0.32) |

A **policy surprise** sends stocks down (-0.98%) and the yield up (+3.4 bp). For a 60/40 plan the stocks cost 0.53 points of funded ratio and the cheaper promise gives 0.34 back: a net of −0.20, inside its band. An **information surprise** of the same size, the Fed revealing good news about the economy, sends stocks and the yield up together, so both channels help (+0.62, also inside its band). A fully matched plan does not move in either case, by construction. So the same rise in rates can hurt or help a pension plan depending on why it happened, and the two channels are of similar size: the net is small and its sign is not settled by these estimates. First order, not a forecast, not pre-registered. `python -m dpc.cli pension-stress`.

Every table is in [reports/pension/results.md](reports/pension/results.md); the page is [shmorsi.github.io/projects/pension-lab.html](https://shmorsi.github.io/projects/pension-lab.html). The code is [src/dpc/pension/](src/dpc/pension/), with its own tests in [tests/test_pension.py](tests/test_pension.py). `python -m dpc.cli pension` downloads the public data, trains the 24 networks and writes every number above; `python -m dpc.cli pension-anatomy` adds the description of what the policy holds.
<!-- pension:end -->

## How it works

**Data.** Kenneth French's 30 value-weighted US industry portfolios, monthly, July 1926 to September 2026 (CRSP vintage 202609). Each portfolio holds every listed firm in its industry at each date, including firms that later failed, so there is no survivorship bias. Industry market caps (number of firms times average size) give the market portfolio; rebuilt from the 30 industries it tracks French's own market return with a correlation of 0.9998. The T-bill rate and the Fama-French factors come from the same library.

**Features, known at each month-end.** For every industry, 13 features: returns over 1, 3, 12 (skipping the last month), 36 and 60 months; volatility over 1 and 5 years; downside deviation and drawdown over 3 years; beta and correlation with the market over 5 years; book-to-market (each year's value, formed at the end of June, used from the end of July); and its weight in the index. Each is compared across the 30 industries (z-scored cross-sectionally, clipped at 4). For the market as a whole, 7 more: its 1- and 5-year return, volatility, 3-year drawdown, the T-bill rate, the spread of industry returns and the market's book-to-market, each z-scored against its own past only.

**The network** ([`src/dpc/wealth/model.py`](src/dpc/wealth/model.py), 20,066 parameters). Each industry is a token (its features plus where its current holding sits against the market), with one more token for the market. Two pre-norm transformer blocks (d = 32, 4 heads) let every industry attend to every other. Two heads read the result:

* **Tilt.** Target weights `softmax(log w_mkt + log 3 · tanh(score))`: each industry's market weight is scaled by a factor between a third and three, then the portfolio is rescaled to 100%. Long-only and fully invested by construction. The tilt head starts at zero, so the untrained network holds exactly the market and must earn every deviation on held-out data.
* **Trade fraction.** κ in (0, 1) from the market token: new holdings are `w + κ (target − w)`. κ = 0 is buy and hold, κ = 1 a full rebalance; how much to trade is learned, not set by hand.

**The loss** ([`simulate`](src/dpc/wealth/model.py)). A differentiable copy of the backtest engine plays each policy forward over 60-month episodes: holdings drift with prices, every unit traded costs 10 bps, and the loss is minus the average monthly log growth of wealth after costs (the Kelly criterion). Gradients flow back through every month of compounding, as in deep hedging, so a trade is judged by the years after it. A test checks that this PyTorch simulator reproduces the NumPy engine month by month.

**Walk-forward training** ([`src/dpc/wealth/train.py`](src/dpc/wealth/train.py)). Refits every five years at the end of December, 1959 to 2024, on data before the refit only. Inside each refit the last 10 years are validation (early stopping on validation growth) and no training label is dated after the first validation decision (asserted in code and tested). Five seeds (42 to 46), each warm-started from its own previous refit; the deployed policy averages their targets and trade fractions. AdamW, gradient clipping, batches of 32 episodes.

**The classical strategies** ([`src/dpc/wealth/baselines.py`](src/dpc/wealth/baselines.py)), fixed before the network was run: the market; equal weight and risk parity, each January; minimum variance (Ledoit-Wolf, at most 20% per industry); a classical growth optimiser with the network's own goal and bounds (maximise μ'w − w'Σw/2 within a third and three times the market weights, plug-in estimates); and a momentum tilt within the same bounds, monthly.

## Quick start

**See the results in a minute**, with no download, no training and no PyTorch:

```bash
pip install -r requirements-dashboard.txt
```

```bash
pip install -e .
```

```bash
python -m dpc.cli quickstart
```

`quickstart` prints the headline table and opens a Streamlit dashboard (growth of $1, the network's holdings against the market's at any month, rolling 10-year growth, every table). It reads only committed results.

**Understand it from first principles:** [`notebooks/02_wealth_from_first_principles.ipynb`](notebooks/02_wealth_from_first_principles.ipynb) shows why growth is not average return, finds the Kelly mix by gradient ascent in PyTorch, shows why trading costs teach a policy to trade less, then reads the real results. It runs in a few minutes on a laptop.

**Reproduce everything from raw data** (about two hours on a 4-core laptop CPU):

```bash
pip install -r requirements.txt
```

```bash
python -m dpc.cli wealth
```

Or stage by stage:

```bash
python -m dpc.cli wealth-data        # download French's files, build the panel, check it against French's market
```

```bash
python -m dpc.cli wealth-baselines   # the six classical strategies
```

```bash
python -m dpc.cli wealth-train --variants all   # the network (5 seeds) and the three ablations (2 seeds each)
```

```bash
python -m dpc.cli wealth-reference   # the network averaged over the ablations' two seeds
```

```bash
python -m dpc.cli wealth-benchmarks  # S&P 500 and NASDAQ Composite monthly returns (added after the test)
```

```bash
python -m dpc.cli wealth-schedules   # the same network rebalanced daily, monthly, quarterly, yearly (added after the test)
```

```bash
python -m dpc.cli wealth-evaluate    # metrics, pre-registered comparisons, decades, costs, seeds
```

```bash
python -m dpc.cli wealth-site        # figures, reports/wealth/results.md and the website export
```

```bash
python scripts/write_readme.py       # this README, from the export
```

```bash
pytest
```

Every setting lives in [`config.yaml`](config.yaml) (section `wealth`), and the loader rejects unknown or missing keys.

## Look-ahead safeguards

* **Point-in-time features.** Features at any date are identical whether or not later data exists (tested by truncating the panel). Book-to-market is used only from the July after it is formed; market caps are those known at the decision.
* **Labels end before validation.** No training episode's last return is dated after the first validation decision (asserted in code and tested).
* **One simulator.** The PyTorch training simulator reproduces the NumPy backtest engine month by month (tested), so the network is trained on exactly the rules it is scored on.
* **Pre-registration.** [`docs/wealth_protocol.md`](docs/wealth_protocol.md) fixed the question, the comparisons and the decision rule in git before the network was run out of sample. The classical strategies' parameters were fixed before they were run.
* **Calibration.** A deliberately cheating policy that sees next month's returns, inside the same bounds and costs, turns $1 into $729 million: the simulator pays for real foresight, so an honest result should look nothing like that.

## Limitations

* **Industries, not funds.** An investor in 1960 could not buy these 30 portfolios cheaply; sector funds came decades later. Costs are a flat 10 bps per unit traded, with no taxes or market impact. A taxable investor also pays tax on realised gains, which favours trading even less.
* **The features carry hindsight.** Momentum, value and low volatility were documented by finance research long after 1960, so the network was handed inputs an investor then would not have known to look at. The pre-registration fixes the test, not the feature list.
* **One country, one century.** The US market from 1960 was among the best in the world, and five seeds of one network on one history is a small sample. Several strategies were compared; the pre-registered primary test is the one to read.

## The first study (archived)

This repository began as a market-neutral long-short study (a transformer return model, a transformer covariance model and end-to-end training through a differentiable optimiser, on 99 large caps since 2008). It is archived in [`docs/longshort.md`](docs/longshort.md), with its research note under [`reports/archive/`](reports/archive/); its code, notebook (`notebooks/01_first_principles.ipynb`) and dashboard (`dashboard/longshort_app.py`) still run.
