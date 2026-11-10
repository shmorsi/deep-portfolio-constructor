"""Write reports/pension/site_pension.json: every figure the website shows for this study, as the text it shows.

The website (shmorsi.github.io) reads only this file for the pension study, so a number on the site
can always be traced to reports/pension/summary.json, which `python -m dpc.cli pension-run` writes.
"""

from __future__ import annotations

import json
from pathlib import Path

LEARNED, GLIDE, SIXTY, LDI, MATCHED = "Learned policy", "Glide path", "60/40", "Liability-driven 40/60", "Fully matched"


def _pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def _signed_pct(x: float, digits: int = 1) -> str:
    return f"{'+' if x > 0 else '−' if x < 0 else ''}{abs(100 * x):.{digits}f}%"


def _cost(x: float) -> str:
    """A cost as dollars per $100 of promises."""
    return f"${100 * x:.1f}"


def export(root: Path) -> dict:
    src = root / "reports" / "pension"
    s = json.loads((src / "summary.json").read_text())
    fans = json.loads((src / "fans.json").read_text())
    sim, hist = s["simulated"], s["history"]
    P, H = sim["policies"], hist["policies"]
    vs_glide, vs_6040 = sim["learned_minus"][GLIDE], sim["learned_minus"][SIXTY]
    where = "deep-portfolio-constructor reports/pension/summary.json"
    stat = lambda label, display: {"label": label, "display": display, "source": where}  # noqa: E731
    stats = {
        "costLearned": stat("What the plan costs its sponsor over 20 years, per $100 promised, learned policy (average of simulated futures)", _cost(P[LEARNED]["mean_cost"])),
        "costGlide": stat("The same, glide path", _cost(P[GLIDE]["mean_cost"])),
        "cost6040": stat("The same, 60/40", _cost(P[SIXTY]["mean_cost"])),
        "costLdi": stat("The same, liability-driven 40/60", _cost(P[LDI]["mean_cost"])),
        "costMatched": stat("The same, fully matched", _cost(P[MATCHED]["mean_cost"])),
        "tailLearned": stat("Cost in the worst 5% of simulated futures, per $100 promised, learned policy", _cost(P[LEARNED]["cvar_cost"])),
        "tailGlide": stat("The same, glide path", _cost(P[GLIDE]["cvar_cost"])),
        "tail6040": stat("The same, 60/40", _cost(P[SIXTY]["cvar_cost"])),
        "tailLdi": stat("The same, liability-driven 40/60", _cost(P[LDI]["cvar_cost"])),
        "tailMatched": stat("The same, fully matched", _cost(P[MATCHED]["cvar_cost"])),
        "fundedLearned": stat("Share of simulated futures fully funded after 20 years, learned policy", _pct(P[LEARNED]["funded_end_share"], 0)),
        "fundedGlide": stat("The same, glide path", _pct(P[GLIDE]["funded_end_share"], 0)),
        "funded6040": stat("The same, 60/40", _pct(P[SIXTY]["funded_end_share"], 0)),
        "fundedLdi": stat("The same, liability-driven 40/60", _pct(P[LDI]["funded_end_share"], 0)),
        "fundedMatched": stat("The same, fully matched", _pct(P[MATCHED]["funded_end_share"], 0)),
        "stocksLearned": stat("Average share in stocks, learned policy", _pct(P[LEARNED]["stocks_mean"], 0)),
        "stocksGlide": stat("Average share in stocks, glide path", _pct(P[GLIDE]["stocks_mean"], 0)),
        "vsGlide": stat("Score of the learned policy minus the glide path (average cost plus worst-5% cost, per $100 promised)", _signed_pct(vs_glide["difference"], 1).replace("%", "")),
        "vsGlideCi": stat("95% bootstrap interval of that difference", f"{_signed_pct(vs_glide['lo'], 1).replace('%', '')} to {_signed_pct(vs_glide['hi'], 1).replace('%', '')}"),
        "vsGlideVerdict": stat("Pre-registered verdict against the glide path, simulated futures", vs_glide["verdict"]),
        "vs6040Verdict": stat("Pre-registered verdict against 60/40, simulated futures", vs_6040["verdict"]),
        "walkLearned": stat("Average cost when yields never come back, per $100 promised, learned policy (not retrained)", _cost(s["model_risk"]["policies"][LEARNED]["mean_cost"])),
        "walkGlide": stat("The same, glide path", _cost(s["model_risk"]["policies"][GLIDE]["mean_cost"])),
        "walkTailLearned": stat("Cost in the worst 5% when yields never come back, learned policy", _cost(s["model_risk"]["policies"][LEARNED]["cvar_cost"])),
        "walkTailGlide": stat("The same, glide path", _cost(s["model_risk"]["policies"][GLIDE]["cvar_cost"])),
        "walkVerdict": stat("Pre-registered verdict against the glide path when yields never come back", s["model_risk"]["learned_minus"][GLIDE]["verdict"]),
        "histLearned": stat(f"Average cost over the {hist['starts']} real 20-year periods, per $100 promised, learned policy", _cost(H[LEARNED]["mean_cost"])),
        "histGlide": stat("The same, glide path", _cost(H[GLIDE]["mean_cost"])),
        "hist6040": stat("The same, 60/40", _cost(H[SIXTY]["mean_cost"])),
        "histVerdict": stat("Pre-registered verdict against the glide path, real history", hist["learned_minus"][GLIDE]["verdict"]),
        "scenarios": stat("Simulated 20-year futures in the test", f"{sim['scenarios']:,}"),
        "starts": stat("Real 20-year periods walked through", str(hist["starts"])),
        "months": stat("Months of data", str(s["data"]["months"])),
        "duration": stat("Liability duration at a 4% yield, years", f"{s['plan']['liability_duration_at_4pct']:.1f}"),
        "fundedStart": stat("Funded ratio at the start", _pct(s["plan"]["funded_at_start"], 0)),
        "networks": stat("Networks trained", str(s["training"]["networks"])),
        "params": stat("Parameters in one network", f"{s['training']['parameters']:,}"),
    }
    out = {"source": f"deep-portfolio-constructor {s['commit']}: reports/pension/summary.json and fans.json", "commit": s["commit"],
           "stats": stats, "fans": fans, "summary": s}
    (src / "site_pension.json").write_text(json.dumps(out, indent=1) + "\n")
    return out
