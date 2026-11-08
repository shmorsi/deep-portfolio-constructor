"""One chart style for every figure in the report (light surface, PNG).

Colors follow a validated categorical order (assigned by entity, never by
rank), recessive grid and axes, thin 2px lines, and a single y-axis per chart.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CONTEXT_GRAY = "#b9b8b0"  # de-emphasised series (context, not identity)
DIVERGING_NEG, DIVERGING_MID, DIVERGING_POS = "#e34948", "#f0efec", "#2a78d6"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "lines.linewidth": 2.0,
            "legend.frameon": False,
            "legend.labelcolor": INK_2,
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.prop_cycle": matplotlib.cycler(color=CATEGORICAL),
        }
    )


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# Dark surface for figures embedded in the (black) website. Same hues, each
# stepped for a dark surface as the validated palette specifies.
DARK = {"surface": "#000000", "ink": "#f5f5f7", "ink_2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "baseline": "#383835"}
LIGHT_TO_DARK = {
    "#2a78d6": "#3987e5", "#eb6834": "#d95926", "#1baf7a": "#199e70", "#eda100": "#c98500",
    "#e87ba4": "#d55181", "#008300": "#008300", "#6250d6": "#9085e9", "#e34948": "#e66767",
    "#52514e": "#c3c2b7", "#898781": "#898781", "#b9b8b0": "#5f5e5a",
}


def apply_dark_style() -> None:
    apply_style()
    plt.rcParams.update({
        "figure.facecolor": DARK["surface"], "axes.facecolor": DARK["surface"], "savefig.facecolor": DARK["surface"],
        "axes.edgecolor": DARK["baseline"], "axes.labelcolor": DARK["ink_2"], "axes.titlecolor": DARK["ink"],
        "grid.color": DARK["grid"], "xtick.color": DARK["muted"], "ytick.color": DARK["muted"],
        "legend.labelcolor": DARK["ink_2"], "text.color": DARK["ink"],
    })


def dark(color: str) -> str:
    return LIGHT_TO_DARK.get(color, color)
