"""Private assets: the cash flows of a fund, pacing a programme, the denominator effect, desmoothing.

**Cash flows (Takahashi and Alexander 2002, the "Yale model").** A fund with commitment CC calls
a fraction RC_t of what is still uncalled, grows its NAV at G, and pays out a fraction RD_t of it:

    C_t   = RC_t (CC - paid-in so far)
    D_t   = RD_t NAV_{t-1} (1 + G),    RD_t = max(Y, (t / L)^B)
    NAV_t = NAV_{t-1} (1 + G) + C_t - D_t

B (the "bow") delays distributions; at t = L everything is paid out and nothing more is called. Parameters here are
illustrative of a buyout fund, not fitted to any manager's data.

**Pacing.** A plan cannot buy a 25% private allocation: it has to commit, wait for calls, and
keep committing as funds pay back. `pace` simulates a plan that commits a share of its assets
each year and finds the share at which the allocation settles at its target; the path there
takes years.

**The denominator effect.** When public markets fall, the private NAV (marked slowly) does not,
so the private share jumps above target and the plan has to stop committing exactly when
vintages are cheapest.

**Desmoothing (Geltner 1993).** Appraisal-based returns are smoothed,
r_obs_t = (1 - phi) r_t + phi r_obs_{t-1}, which understates volatility and correlation. With
phi estimated as the first-order autocorrelation of r_obs, r_t = (r_obs_t - phi r_obs_{t-1}) / (1 - phi).
`desmoothing_check` smooths a series with known volatility and shows how much is recovered.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FundModel:
    life: int = 12
    bow: float = 2.5
    growth: float = 0.11
    yield_: float = 0.0
    calls: tuple = (0.25, 0.333, 0.5)      # then 0.5 a year on what is left

    def rc(self, t: int) -> float:
        return self.calls[t - 1] if t <= len(self.calls) else self.calls[-1]

    def rd(self, t: int) -> float:
        return max(self.yield_, (t / self.life) ** self.bow)

    def simulate(self, commitment: float = 1.0, growth: np.ndarray | None = None) -> dict:
        """Yearly calls, distributions and NAV per unit committed. `growth` overrides G year by year."""
        L = self.life
        C, D, N = np.zeros(L + 1), np.zeros(L + 1), np.zeros(L + 1)
        paid = 0.0
        for t in range(1, L + 1):
            g = self.growth if growth is None else growth[t - 1]
            C[t] = self.rc(t) * (commitment - paid) if t < L else 0.0   # nothing is called in the liquidation year
            paid += C[t]
            D[t] = min(self.rd(t), 1.0) * N[t - 1] * (1 + g)
            N[t] = N[t - 1] * (1 + g) + C[t] - D[t]
        return {"calls": C, "distributions": D, "nav": N, "paid_in": paid}


def tvpi_irr(sim: dict) -> tuple[float, float]:
    net = sim["distributions"] - sim["calls"]
    net[-1] += sim["nav"][-1]
    tvpi = float((sim["distributions"].sum() + sim["nav"][-1]) / sim["calls"].sum())
    # IRR by bisection on the yearly net cash flows
    lo, hi = -0.99, 1.0
    f = lambda r: float(np.sum(net / (1 + r) ** np.arange(len(net))))
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return tvpi, 0.5 * (lo + hi)


def pace(rate: float, years: int = 30, public_return: np.ndarray | float = 0.06, fund: FundModel = FundModel(),
         start_assets: float = 100.0, net_outflow: float = 0.0, dynamic_target: float | None = None) -> dict:
    """Commit `rate` x total assets each year; with `dynamic_target`, commit less (down to nothing) while over target.

    The rule only brakes. A rule that also commits more when under target overshoots, because calls
    arrive years after commitments: the classic pacing mistake.

    Public assets earn `public_return` (a number or a yearly path); private NAV follows the fund model.
    Calls are paid out of public assets, distributions are reinvested in them. `net_outflow` is the
    benefit payments net of contributions, as a share of total assets.
    """
    pr = np.full(years, public_return) if np.isscalar(public_return) else np.asarray(public_return, float)
    L = fund.life
    public = start_assets
    vintages = []          # (start year, commitment)
    share, total_path, commits, nav_path, net_cf = [], [], [], [], []
    rc = [fund.rc(t) for t in range(1, L + 1)]
    for y in range(years):
        # existing vintages step forward one year
        nav = 0.0
        calls = dists = 0.0
        new_v = []
        for (s, c, paid, n) in vintages:
            t = y - s + 1
            if t > L:
                continue
            call = rc[t - 1] * (c - paid) if t < L else 0.0
            dist = min(fund.rd(t), 1.0) * n * (1 + fund.growth)
            n1 = n * (1 + fund.growth) + call - dist
            calls += call
            dists += dist
            nav += n1
            new_v.append((s, c, paid + call, n1))
        vintages = new_v
        public = public * (1 + pr[y]) - calls + dists
        total = public + nav
        public -= net_outflow * total
        total = public + nav
        cur = nav / total
        r = rate
        if dynamic_target is not None and cur > dynamic_target:      # brake only: never accelerate (calls lag commitments)
            r = rate * max(0.0, 1.0 - 3.0 * (cur - dynamic_target) / dynamic_target)
        commit = r * total
        vintages.append((y + 1, commit, 0.0, 0.0))
        share.append(cur)
        total_path.append(total)
        commits.append(commit / total)
        nav_path.append(nav)
        net_cf.append((dists - calls) / total)
    return {"share": np.array(share), "total": np.array(total_path), "commit_rate": np.array(commits),
            "nav": np.array(nav_path), "net_cash_flow": np.array(net_cf)}


def rate_for_target(target: float, **kw) -> float:
    """The constant commitment rate at which the private share settles (the average of the last 5 years) at target."""
    lo, hi = 0.0, 0.5
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        s = pace(mid, **kw)["share"][-5:].mean()
        lo, hi = (mid, hi) if s < target else (lo, mid)
    return 0.5 * (lo + hi)


def geltner_unsmooth(r_obs: np.ndarray) -> tuple[np.ndarray, float]:
    r = np.asarray(r_obs, float)
    phi = float(np.corrcoef(r[1:], r[:-1])[0, 1])
    return (r[1:] - phi * r[:-1]) / (1 - phi), phi


def smooth(r_true: np.ndarray, phi: float) -> np.ndarray:
    out = np.empty_like(r_true, dtype=float)
    out[0] = r_true[0]
    for t in range(1, len(r_true)):
        out[t] = (1 - phi) * r_true[t] + phi * out[t - 1]
    return out


def desmoothing_check(r_true: np.ndarray, phi: float, other: np.ndarray | None = None) -> dict:
    """Smooth a series whose volatility is known, unsmooth it, and compare (and, if given, its correlation with `other`)."""
    obs = smooth(r_true, phi)
    rec, phi_hat = geltner_unsmooth(obs)
    out = {"phi": phi, "phi_hat": phi_hat, "vol_true": float(np.std(r_true[1:], ddof=1)),
           "vol_observed": float(np.std(obs[1:], ddof=1)), "vol_desmoothed": float(np.std(rec, ddof=1)),
           "corr_true_desmoothed": float(np.corrcoef(r_true[1:], rec)[0, 1])}
    if other is not None:
        out["corr_with_other_true"] = float(np.corrcoef(r_true[1:], other[1:])[0, 1])
        out["corr_with_other_observed"] = float(np.corrcoef(obs[1:], other[1:])[0, 1])
        out["corr_with_other_desmoothed"] = float(np.corrcoef(rec, other[1:])[0, 1])
    return out
