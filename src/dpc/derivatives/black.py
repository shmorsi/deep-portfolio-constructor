"""Closed-form option prices and Greeks.

Black-Scholes-Merton (spot S, continuous dividend yield q):
    d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt T),  d2 = d1 - sigma sqrt T
    call = S e^{-qT} N(d1) - K e^{-rT} N(d2)
Black-76 is the same with the forward F in place of S e^{(r-q)T}, everything discounted at r:
the model for options on futures, caps, floors and swaptions.
Bachelier (normal) prices with an absolute volatility, the convention for rates since
negative rates made lognormal models break.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d1d2(S, K, T, r, q, sigma):
    v = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / v
    return d1, d1 - v


def bsm(S, K, T, r, sigma, q=0.0, kind="call"):
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    if kind == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def greeks(S, K, T, r, sigma, q=0.0, kind="call") -> dict[str, float]:
    """Delta, gamma, vega (per 1.00 of vol), theta (per year), rho (per 1.00 of rate)."""
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    disc_q, disc_r = math.exp(-q * T), math.exp(-r * T)
    pdf = norm.pdf(d1)
    gamma = disc_q * pdf / (S * sigma * math.sqrt(T))
    vega = S * disc_q * pdf * math.sqrt(T)
    if kind == "call":
        delta = disc_q * norm.cdf(d1)
        theta = -S * disc_q * pdf * sigma / (2 * math.sqrt(T)) - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1)
        rho = K * T * disc_r * norm.cdf(d2)
    else:
        delta = -disc_q * norm.cdf(-d1)
        theta = -S * disc_q * pdf * sigma / (2 * math.sqrt(T)) + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)
        rho = -K * T * disc_r * norm.cdf(-d2)
    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega), "theta": float(theta), "rho": float(rho)}


def black76(F, K, T, r, sigma, kind="call"):
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    df = np.exp(-r * T)
    if kind == "call":
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def bachelier(F, K, T, r, sigma_n, kind="call"):
    """Normal model: sigma_n is an absolute volatility (e.g. 0.01 = 100 bp a year)."""
    v = sigma_n * np.sqrt(T)
    d = (F - K) / v
    df = np.exp(-r * T)
    if kind == "call":
        return df * ((F - K) * norm.cdf(d) + v * norm.pdf(d))
    return df * ((K - F) * norm.cdf(-d) + v * norm.pdf(d))


def implied_vol(price, S, K, T, r, q=0.0, kind="call", lo=1e-6, hi=5.0) -> float:
    """The sigma that reproduces `price` (Brent's method); NaN outside the no-arbitrage bounds."""
    intrinsic = max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0) if kind == "call" else max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0)
    upper = S * math.exp(-q * T) if kind == "call" else K * math.exp(-r * T)
    if not intrinsic < price < upper:
        return float("nan")
    return float(brentq(lambda s: bsm(S, K, T, r, s, q, kind) - price, lo, hi, xtol=1e-12))
