"""Pension Lab: can a pension plan keep its promise funded?

A stylised defined-benefit plan: a 60-year schedule of inflation-indexed payments (the
liability), valued off the government bond yield, against assets held in stocks, 10-year
bonds and a bond portfolio that matches the liability. The plan starts 90% funded and its
sponsor tops it up when it falls short. The study asks what that costs the sponsor over 20
years under four standard policies and one learned through the same differentiable
simulation idea as the wealth study, on real history and on simulated futures.

Modules, in the order the data flows:

    data.py          the monthly panel: equity returns, the 10-year yield, inflation
    fixed_income.py  bond returns, present value, duration and convexity
    liability.py     the payment schedule and its value along a scenario
    scenarios.py     the scenario generator (AR(1) yields and inflation, block-bootstrapped shocks)
    plan.py          the plan simulator: assets, benefits, contributions, funded ratio
    policy.py        60/40, liability-driven, fully matched, a glide path, and the learned policy
    train.py         training the learned policy through the simulator
    evaluate.py      the sponsor's cost, tail risk, and the pre-registered comparisons
    run.py           the whole study; site.py writes what the website reads

The protocol (every constant, comparison and decision rule, fixed before the test) is
docs/pension_protocol.md.
"""
