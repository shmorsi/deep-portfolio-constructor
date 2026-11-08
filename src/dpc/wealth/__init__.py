"""Long-only, long-horizon wealth study.

A neural network decides how much of a fully invested, long-only portfolio to
hold in each of 30 US industries, month by month since 1960, and is trained
on the quantity a long-term investor cares about: the growth rate of wealth
after trading costs. Data are Kenneth French's industry portfolios, 1926 to
today, which have no survivorship bias.

Modules
-------
data       download, parse and cache the industry and factor files
features   point-in-time features for every decision date
engine     the monthly investing simulator: prices drift holdings, trades cost
baselines  the classical strategies every result is compared with
model      the Deep Growth Network policy
train      walk-forward training through the differentiable simulator
evaluate   wealth, risk, decade and significance statistics
report     tables, figures and the site export
"""
