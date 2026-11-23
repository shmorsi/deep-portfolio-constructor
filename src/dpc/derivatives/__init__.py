"""Derivatives and rates: what a capital-markets desk and a pension's overlay team price every day.

    black       Black-Scholes-Merton, Black-76 and Bachelier prices, Greeks, implied volatility
    trees       Cox-Ross-Rubinstein binomial trees for American options
    montecarlo  Monte Carlo with antithetic and control variates (Asian options)
    sabr        Hagan's SABR smile and its calibration
    rates       bootstrapping a zero curve from par yields, interest-rate swaps, DV01 and key-rate risk
    hedging     delta hedging: discrete-rebalancing error and the volatility risk premium on real data

Every formula is checked against a textbook value or an identity in tests/test_derivatives.py.
"""
