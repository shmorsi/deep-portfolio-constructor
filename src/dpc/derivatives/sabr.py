"""SABR (Hagan, Kumar, Lesniewski and Woodward 2002): the smile model of rates and FX desks.

dF = alpha F^beta dW1,  dalpha = nu alpha dW2,  dW1 dW2 = rho dt.
Hagan's expansion gives a Black implied volatility for each strike. With beta fixed (here 0.5,
a common choice for rates), alpha sets the level, rho the skew and nu the curvature; the three
are fitted by least squares to a quoted smile.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def implied_vol(F, K, T, alpha, beta, rho, nu):
    F, K = float(F), np.asarray(K, float)
    one = 1 - beta
    fk = F * K
    logfk = np.log(F / K)
    z = nu / alpha * fk ** (one / 2) * logfk
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))
        zx = np.where(np.abs(z) < 1e-8, 1.0, z / x)
    denom = fk ** (one / 2) * (1 + one ** 2 / 24 * logfk ** 2 + one ** 4 / 1920 * logfk ** 4)
    corr = 1 + (one ** 2 / 24 * alpha ** 2 / fk ** one + rho * beta * nu * alpha / (4 * fk ** (one / 2))
                + (2 - 3 * rho ** 2) / 24 * nu ** 2) * T
    return alpha / denom * zx * corr


def calibrate(F, T, strikes, vols, beta=0.5) -> dict[str, float]:
    strikes, vols = np.asarray(strikes, float), np.asarray(vols, float)
    atm = float(np.interp(F, strikes, vols))
    x0 = [atm * F ** (1 - beta), -0.2, 0.5]

    def resid(p):
        return implied_vol(F, strikes, T, p[0], beta, p[1], p[2]) - vols

    fit = least_squares(resid, x0, bounds=([1e-6, -0.999, 1e-4], [10.0, 0.999, 10.0]))
    a, rho, nu = fit.x
    return {"alpha": float(a), "beta": beta, "rho": float(rho), "nu": float(nu),
            "rmse": float(np.sqrt(np.mean(fit.fun ** 2)))}
