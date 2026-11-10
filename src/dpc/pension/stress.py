"""What a Fed surprise does to the plan: the third study's estimates, applied to the second study's plan.

World Shock Lab (github.com/shmorsi/world-shock-lab) measures what a surprise in US monetary policy
does to stocks and to the 10-year yield, per 10 basis points of surprise, with standard errors. This
module turns those two numbers into the one a pension plan cares about: the move in its funded ratio.

    stocks move       -> the plan's stocks move with them
    the yield moves   -> its bonds are repriced, and so is the promise it owes

A plan in bonds matched to the promise does not move at all. A plan in stocks is hit twice, or helped
twice, depending on why rates rose: a policy surprise sends stocks down and yields up; an information
surprise (the Fed revealing good news about the economy) sends both up.

First order, on the announcement day and 20 trading days later, at the plan's starting position
(90% funded) and a 4% yield. Not a forecast, and not part of the pre-registered tests.

    extract(lab)   copies the estimates it needs from the lab's committed results, with their commit
    stress(root)   writes reports/pension/stress.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .fixed_income import par_bond_return
from .plan import Plan
from .policy import Ensemble, Glide, Learned, baselines

SHOCKS = (("fed_mp", "Fed policy surprise"), ("fed_cbi", "Fed information surprise"))
OUTCOMES = {"stocks": "SPY.tr", "yield": "US10Y"}      # the lab's ids: S&P 500 ETF total return (%), 10-year Treasury yield (bp)
HORIZONS = (0, 20)
YIELD = 0.04                                           # the yield the plan's duration is quoted at
Z90 = 1.645
EXTRACT = "world_shock_lab_extract.json"


def extract(lab: Path, out_dir: Path) -> dict:
    """Copy the estimates this module uses out of the lab's committed catalog, with where they came from."""
    catalog = json.loads((lab / "web" / "public" / "data" / "catalog.json").read_text())
    meta = {s["id"]: s for s in catalog["shocks"]}
    out = {"source": "world-shock-lab web/public/data/catalog.json", "lab_commit": catalog["provenance"]["git_sha"],
           "generated_at": catalog["provenance"]["generated_at"], "shocks": {}}
    for sid, _ in SHOCKS:
        out["shocks"][sid] = {"name": meta[sid]["name"], "per": meta[sid]["per"], "unit": meta[sid]["unit"], "label": meta[sid]["label"],
                              **{key: {"coef": catalog["irfs"][sid][oid]["coef"], "se": catalog["irfs"][sid][oid]["se"], "n": catalog["irfs"][sid][oid]["n"],
                                       "units": catalog["irfs"][sid][oid]["units"]} for key, oid in OUTCOMES.items()}}
    (out_dir / EXTRACT).write_text(json.dumps(out, indent=1) + "\n")
    return out


def _weights(root: Path, plan: Plan) -> dict[str, np.ndarray]:
    """What each policy holds at the plan's starting position (the glide path and the network look at the funded ratio)."""
    state = Learned.CENTRE.repeat(1, 1).clone()
    state[0, 0], state[0, 1], state[0, 6] = plan.funded0, YIELD, 0.0
    with torch.no_grad():
        state[0, 7:10] = Glide()(state)[0]
        held = {name: policy(state)[0].numpy().astype(float) for name, policy in baselines().items()}
        saved = root / "reports" / "pension" / "learned_policy.pt"
        if saved.is_file():                                  # the scored networks (anatomy.py saves them)
            nets = []
            for weights in torch.load(saved).values():
                net = Learned()
                net.load_state_dict(weights)
                nets.append(net.eval())
            held["Learned policy"] = Ensemble(nets)(state)[0].numpy().astype(float)
    return held


def funded_move(w: np.ndarray, d_stocks: float, d_yield: float, plan: Plan) -> float:
    """The change in the funded ratio, in points, when stocks return `d_stocks` and the yield moves by `d_yield`."""
    bond = float(par_bond_return(YIELD, YIELD + d_yield) - par_bond_return(YIELD, YIELD))
    promise = float(plan.schedule.unit_value(YIELD + d_yield) / plan.schedule.unit_value(YIELD) - 1.0)
    assets = w[0] * d_stocks + w[1] * bond + w[2] * promise
    return 100.0 * plan.funded0 * ((1.0 + assets) / (1.0 + promise) - 1.0)


def stress(root: Path, log=print) -> dict:
    out_dir = root / "reports" / "pension"
    lab = json.loads((out_dir / EXTRACT).read_text())
    plan = Plan()
    held = _weights(root, plan)
    result = {"note": "First-order effect of a Fed surprise on the plan's funded ratio, from World Shock Lab's estimates. "
                      "Added after the test; not pre-registered; not a forecast.",
              "lab": {k: lab[k] for k in ("source", "lab_commit", "generated_at")},
              "at": {"funded": plan.funded0, "yield": YIELD}, "weights": {k: [float(x) for x in v] for k, v in held.items()}, "shocks": {}}
    for sid, name in SHOCKS:
        s = lab["shocks"][sid]
        rows = {}
        for h in HORIZONS:
            ds, ses = s["stocks"]["coef"][h] / 100.0, s["stocks"]["se"][h] / 100.0            # % -> fraction
            dy, sey = s["yield"]["coef"][h] / 1e4, s["yield"]["se"][h] / 1e4                 # bp -> fraction
            policies = {}
            for pname, w in held.items():
                mid = funded_move(w, ds, dy, plan)
                # The two estimates come from the same announcements, so their errors are not independent and the lab does not
                # publish their covariance. Adding the two half-widths is the widest the band can be whatever the correlation.
                half = Z90 * (abs(funded_move(w, ses, 0.0, plan)) + abs(funded_move(w, 0.0, sey, plan)))
                policies[pname] = {"points": mid, "half_width": half, "excludes_zero": bool(abs(mid) > half),
                                   "from_stocks": funded_move(w, ds, 0.0, plan), "from_yield": funded_move(w, 0.0, dy, plan)}
            rows[str(h)] = {"stocks_pct": s["stocks"]["coef"][h], "stocks_se": s["stocks"]["se"][h],
                            "yield_bp": s["yield"]["coef"][h], "yield_se": s["yield"]["se"][h],
                            "promise_pct": 100.0 * float(plan.schedule.unit_value(YIELD + dy) / plan.schedule.unit_value(YIELD) - 1.0),
                            "policies": policies}
        result["shocks"][sid] = {"name": name, "per": s["per"], "unit": s["unit"], "events": s["stocks"]["n"][0], "horizons": rows}
        day0 = rows["0"]
        log(f"  {name}, +{s['per']:g} {s['unit']}: stocks {day0['stocks_pct']:+.2f}%, yield {day0['yield_bp']:+.1f} bp, the promise {day0['promise_pct']:+.2f}%; "
            + ", ".join(f"{p} {v['points']:+.2f}" for p, v in day0["policies"].items()))
    (out_dir / "stress.json").write_text(json.dumps(result, indent=1) + "\n")
    log(f"Wrote {(out_dir / 'stress.json').relative_to(root)}")
    return result
