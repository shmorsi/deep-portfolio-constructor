# Pension Lab: protocol

Fixed before the test was run. This file and the code under `src/dpc/pension/` were committed
first; the results under `reports/pension/` were produced afterwards by `python -m dpc.cli pension`
and committed separately. Anything that had to change after the first run is listed under
[Deviations](#deviations), with the reason.

## The question

A pension plan has promised payments for the next 60 years and holds less than those promises
are worth. Its sponsor has to make up the difference. **Over the next 20 years, what does that
cost the sponsor, how bad can it get, and does a policy learned by simulation do better than
the rules plans actually use?**

## The plan

| | |
|---|---|
| Promise | Monthly payments for 60 years. Each is `exp(-1/480)` times the one before (a mature plan running off). |
| Indexing | Each payment is made in the money of the month it falls due: fully indexed to consumer prices. |
| Value of the promise | Future payments are projected to grow with expected inflation and discounted at the 10-year government yield, monthly compounding (the same as discounting at a real yield). One yield for every maturity (a flat curve). Expected inflation is the average of the last ten years, to the last published month. Scaled to 100 at the start. |
| Assets at the start | 90% of the promise's value. |
| Benefits | Paid out of the assets at the end of each month. |
| Contributions | When assets are below the promise's value, the sponsor pays in 1/120 of the shortfall that month (a ten-year amortisation). A surplus is never paid back. |
| Trading | 10 basis points of every dollar traded. No shorting, no leverage, no derivatives. |
| Horizon | 240 months. |

Three assets:

1. **Stocks**: the US market's total return (Kenneth French's `Mkt-RF + RF`).
2. **10-year bonds**: a constant-maturity series built the usual way: buy a par bond, hold it a
   month, sell it at the new yield with a month less to run (`fixed_income.par_bond_return`).
3. **Matching bonds**: nominal bonds bought each month to pay exactly the schedule as it is then
   expected. They hedge the yield completely. They do not hedge inflation turning out different
   from what was expected, or the expectation itself changing (there is no inflation-linked bond here).

## What is measured

**The sponsor's cost** of a scenario: every contribution over the 20 years, plus the shortfall
still left at the end, in start-of-plan money, per 100 of promises. A surplus at the end counts
for nothing.

**The score** of a policy over a set of scenarios: the average cost, plus the average cost in the
worst 5% of scenarios (expected shortfall at 5%). Lower is better. This one number is both the
training objective and the test statistic.

Also reported, without a test: the share of scenarios fully funded at year 20, the median funded
ratio at year 20, the lowest funded ratio reached, and the average holdings.

## The policies

| Policy | Rule |
|---|---|
| 60/40 | 60% stocks, 40% 10-year bonds, rebalanced monthly. |
| Liability-driven 40/60 | 40% stocks, 60% matching bonds. |
| Fully matched | 100% matching bonds. |
| Glide path | 70% stocks when the funded ratio is 0.80 or less, 20% when it is 1.10 or more, a straight line between; the rest in matching bonds. |
| Learned policy | A network (10 inputs, two hidden layers of 32, `tanh`) whose output is added to the logarithm of the glide path's weights (plus a 2% floor) and passed through a softmax. It starts as the glide path and can only ever be long-only and fully invested. Three networks (seeds 1, 2, 3) are trained and their weights averaged. |

The network sees, at each month-end: the funded ratio; the yield; its change over 12 months; the
stock market's return over the last 12 months; inflation over the 12 months to the last
*published* month (one month late, as CPI is); expected inflation (the ten-year average); the
share of the horizon elapsed; and the weights it holds. It never sees a forecast.

## The scenario generator

Fitted on monthly data and nothing else (`scenarios.py`):

- The yield and monthly inflation each follow an AR(1) fitted by least squares, with the
  persistence capped at 0.995. The yield is floored at 0.10%.
- Stock returns are used as they came.
- The three monthly shocks (stock return, yield shock, inflation shock) are resampled together
  in blocks (stationary bootstrap, average block 12 months).
- Each scenario starts 122 months before the plan does, from a month of history drawn at random,
  so the plan begins with ten years of inflation behind it.

It has no views about the future. It rearranges the past.

## Training

Adam, learning rate 0.01, 300 steps, 1,024 scenarios a step drawn from 8,192; the gradient of
the score flows back through all 240 months of the plan. The network kept is the one with the
best score on 4,096 separate validation scenarios, checked every 30 steps. These settings were
chosen on validation and development scenarios (seed 777); the test scenarios were not generated
until this file was committed.

## The tests

**1. Simulated futures (the main test).** The generator is fitted on all the data. The five
policies are run on 10,000 scenarios (seed 2026) that no training or validation set used.
For the learned policy against each of the four others: the difference in score, with a 95%
bootstrap interval over scenarios (2,000 draws).

**1b. Model risk: yields that never come back.** In the generator above the yield is always
pulled back toward its average, and a policy trained on it can learn to bet on that. So the same
five policies, with the learned one *not retrained*, are also run on 10,000 scenarios (seed
2027) from a generator in which the yield is a random walk: each month's yield is last month's
plus a change that happened in history, with no pull to anything. Same statistic, same interval,
same decision rule. If the learned policy's advantage is a bet on yields returning, it should
shrink or reverse here.

**2. Real history, walking forward.** For each year from 1973 to 2006, a plan starts that
December and lives through the 20 years that actually followed (34 periods, the last ending in
December 2025). The learned policy used for a start year was trained on a generator fitted
only to data up to that start, refitted every five years (1973, 1978, ... 2003). For the learned
policy against each of the four others: the difference in average cost, with a 95% moving-block
bootstrap interval (blocks of 10 consecutive start years).

**Decision rule, for both.** If the whole interval is below zero the learned policy is
**better**; if the whole interval is above zero it is **worse**; otherwise there is **no
detectable difference**. The headline comparison is against the glide path, on test 1.

## What will be said whatever happens

- Every policy's average cost, worst-5% cost and funded share, in one table, including any that
  the learned policy does not improve on.
- Test 1 is run inside the model the learned policy was trained on. A good result there shows the
  policy learned the model, not that the model is the world.
- Test 2 has 34 overlapping periods, which is about three independent ones, all in one country.
  It can show a policy failing; it cannot establish a small advantage.
- If the learned policy is not better than the glide path, the page says so in its first line.

## Limits fixed by the design

- One yield for all maturities: no curve risk, and the matching bonds hedge the promise's value
  perfectly, which no real portfolio does.
- No inflation-linked bonds, so an inflation surprise can only be absorbed, not hedged.
- Expected inflation is the last ten years' average: it moves slowly and can be wrong for a long time.
- No credit, real assets or private assets; no longevity risk; no new members.
- US data. Yields before 1962 are monthly averages, smoother than month-end readings; they only
  feed the generator's early fits. The consumer price index for October 2025 was never published;
  it is interpolated between September and November.
- The discount rate is a government yield. Plans often use a higher one, which makes the same
  promise look smaller.

## Deviations

None. The study ran once, at commit `304f7c8`, and its output is committed as it came out
(`reports/pension/summary.json`; every table in `reports/pension/results.md`).

## After the test

Written after the results above existed. Nothing here was pre-registered, and none of it changes a
verdict. It is in `src/dpc/pension/anatomy.py` (`python -m dpc.cli pension-anatomy`), which retrains
the three networks of test 1 from the same seeds and stops unless they reproduce the committed
result to the last digit; their weights are in `reports/pension/learned_policy.pt`.

The tests said: better in simulation, no detectable difference on real history. These three
additions ask why.

1. **What the policy holds.** Its average weights in the test-1 futures, by funded ratio, by the
   level of the yield and by year, next to the glide path in the same situations.
2. **A tuned glide path.** The protocol's glide path has textbook numbers (70% stocks under 80%
   funded, sliding to 20% at 110%). Here those four numbers are trained on the same scenarios, with
   the same score and steps as the network, and the result is run on the test-1 and test-1b
   futures. It separates "a rule tuned to the cost" from "a network with more inputs".
3. **Futures with nothing to recognise.** The scenario generator resamples history in runs
   averaging 12 months, and the network sees the last 12 months of returns, yields and prices. The
   same policies, not retrained, are run on 10,000 futures whose months are drawn one at a time
   (seed 2028, used by nothing else). If the network's lead survives, it does not come from reading
   runs of history; if it does not, it does.

The readings use the protocol's rule (a 95% bootstrap interval that excludes zero), but an
analysis chosen after seeing a result is weaker evidence than a test fixed before it, and is
labelled as such wherever it is quoted.
