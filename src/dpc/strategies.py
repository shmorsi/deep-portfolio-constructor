"""Strategy names, labels and colours: one source for reports, figures and the dashboard.

Light on purpose (pandas only), so the dashboard can import it without
PyTorch or cvxpy installed.
"""

from __future__ import annotations

from pathlib import Path

LONG_ONLY = ["equal_weight", "min_variance", "mean_variance_lw", "risk_parity", "hrp"]
LONG_SHORT = ["ls_naive", "ls_classical", "ls_deep_twostage", "ls_deep_alpha", "e2e", "e2e_deep"]
FLAGSHIP = "e2e"  # the end-to-end model pre-specified in Stage 5; e2e_deep is a later extension

LABELS = {
    "equal_weight": "Equal weight (LO)",
    "min_variance": "Min variance (LO)",
    "mean_variance_lw": "Mean-variance LW (LO)",
    "risk_parity": "Risk parity (LO)",
    "hrp": "HRP (LO)",
    "ls_naive": "L/S naive signal",
    "ls_classical": "L/S classical (LW + optimiser)",
    "ls_deep_twostage": "L/S deep risk + optimiser",
    "ls_deep_alpha": "L/S deep views + deep risk + optimiser",
    "e2e": "L/S end-to-end (classical views)",
    "e2e_deep": "L/S end-to-end, fully deep",
}

SHORT = {
    "equal_weight": "Equal wt", "min_variance": "Min var", "mean_variance_lw": "MV (LW)", "risk_parity": "Risk parity",
    "hrp": "HRP", "ls_naive": "Naive L/S", "ls_classical": "Classical L/S", "ls_deep_twostage": "Deep risk 2-stage",
    "ls_deep_alpha": "Deep views 2-stage", "e2e": "E2E (classical views)", "e2e_deep": "E2E fully deep",
}

# Colour follows the strategy in every chart (never its rank). Validated categorical order;
# long-only baselines are context, so three of them share muted greys.
COLORS = {
    "e2e": "#2a78d6",  # blue: the primary (pre-specified) end-to-end model
    "e2e_deep": "#6250d6",  # violet: fully deep extension
    "ls_deep_alpha": "#e87ba4",  # magenta
    "ls_deep_twostage": "#eda100",  # yellow
    "ls_classical": "#1baf7a",  # aqua
    "ls_naive": "#eb6834",  # orange
    "equal_weight": "#e34948",  # red
    "hrp": "#008300",  # green
    "risk_parity": "#52514e",
    "min_variance": "#898781",
    "mean_variance_lw": "#b9b8b0",
}


def load_all_results(folder: Path) -> dict:
    """Every saved backtest in ``folder`` that belongs to a known strategy."""
    from dpc.backtest import load_result

    out = {}
    for name in (*LONG_ONLY, *LONG_SHORT):
        if (folder / f"{name}__daily.parquet").exists():
            out[name] = load_result(name, folder)
    return out
