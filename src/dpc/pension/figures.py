"""The pension study's figures, drawn from reports/pension/*.json in the report's one chart style.

  fans.png     the funded ratio over 20 years under 60/40, the glide path and the learned policy
               (the same 10,000 simulated futures in each panel)
  cost.png     what each policy costs the sponsor: on average, and in the worst 5% of futures
  anatomy.png  what the learned policy holds at each funded ratio, next to the glide path it started as
               (only when reports/pension/anatomy.json exists)

Writes reports/figures/pension/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from dpc.plotting import BASELINE, CATEGORICAL, INK, INK_2, MUTED, SURFACE, apply_style, save

LEARNED, GLIDE = "Learned policy", "Glide path"
ORDER = ["60/40", "Liability-driven 40/60", "Fully matched", GLIDE, LEARNED]
BLUE, ORANGE = CATEGORICAL[0], CATEGORICAL[1]
pct = FuncFormatter(lambda v, _: f"{100 * v:.0f}%")


def fans(src: Path, out: Path) -> Path:
    f = json.loads((src / "fans.json").read_text())
    s = json.loads((src / "summary.json").read_text())
    years = np.array(f["years"])
    panels = [("60/40", "60% stocks, 40% bonds"), (GLIDE, "Glide path"), (LEARNED, "Learned policy")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), sharey=True)
    for ax, (name, title) in zip(axes, panels):
        p = {k: np.array(v) for k, v in f["percentiles"][name].items()}
        ax.fill_between(years, p["p05"], p["p95"], color=BLUE, alpha=0.14, lw=0)      # one hue, light to dark: 90% of futures, then the middle half
        ax.fill_between(years, p["p25"], p["p75"], color=BLUE, alpha=0.30, lw=0)
        ax.plot(years, p["p50"], color=BLUE, lw=2, solid_capstyle="round")
        ax.axhline(1.0, color=INK_2, lw=1)
        ax.set_yscale("log")
        ax.set_yticks([0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
        ax.yaxis.set_major_formatter(pct)
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_xlim(0, 20)
        ax.grid(axis="x", visible=False)
        row = s["simulated"]["policies"][name]
        ax.set_title(title, pad=22)
        ax.text(0.0, 1.025, f"fully funded at year 20 in {100 * row['funded_end_share']:.0f}% of futures", transform=ax.transAxes,
                va="bottom", ha="left", fontsize=8.5, color=INK_2)
        ax.set_xlabel("Years")
    lo = min(min(f["percentiles"][n]["p05"]) for n, _ in panels)
    hi = max(max(f["percentiles"][n]["p95"]) for n, _ in panels)
    axes[0].set_ylim(max(0.35, lo * 0.92), hi * 1.08)
    axes[0].set_ylabel("Funded ratio (assets ÷ what is owed)")
    axes[0].text(20, 1.0, "fully funded ", ha="right", va="bottom", fontsize=8, color=INK_2)
    axes[2].text(20.3, f["percentiles"][LEARNED]["p50"][-1], "median", va="center", fontsize=8, color=INK_2, clip_on=False)
    axes[2].text(20.3, f["percentiles"][LEARNED]["p95"][-1], "95th\npercentile", va="center", fontsize=8, color=MUTED, clip_on=False)
    axes[2].text(20.3, f["percentiles"][LEARNED]["p05"][-1], "5th", va="center", fontsize=8, color=MUTED, clip_on=False)
    fig.suptitle(f"A pension plan that starts {100 * s['plan']['funded_at_start']:.0f}% funded: {s['simulated']['scenarios']:,} simulated 20-year futures",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK, y=1.08)
    fig.subplots_adjust(wspace=0.08)
    path = out / "fans.png"
    save(fig, path)
    return path


def cost(src: Path, out: Path) -> Path:
    s = json.loads((src / "summary.json").read_text())
    P = s["simulated"]["policies"]
    names = ORDER[::-1]
    y = np.arange(len(names))
    mean = np.array([100 * P[n]["mean_cost"] for n in names])
    tail = np.array([100 * P[n]["cvar_cost"] for n in names])
    fig, ax = plt.subplots(figsize=(9, 3.9))
    h = 0.32
    ax.barh(y + h / 2 + 0.02, mean, height=h, color=BLUE, label="Average of all futures")
    ax.barh(y - h / 2 - 0.02, tail, height=h, color=ORANGE, label="Average of the worst 5% of futures")
    for yi, m, t in zip(y, mean, tail):
        ax.text(m + tail.max() * 0.012, yi + h / 2 + 0.02, f"${m:.1f}", va="center", fontsize=9, color=INK)
        ax.text(t + tail.max() * 0.012, yi - h / 2 - 0.02, f"${t:.1f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, names)
    for label in ax.get_yticklabels():
        label.set_color(INK if label.get_text() == LEARNED else INK_2)
        label.set_fontweight("bold" if label.get_text() == LEARNED else "normal")
    ax.set_xlim(0, tail.max() * 1.13)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:.0f}"))
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_color(BASELINE)
    ax.set_xlabel("Cost to the sponsor over 20 years, per $100 promised (contributions plus any shortfall left)")
    ax.set_title(f"What each policy costs the plan's sponsor, {s['simulated']['scenarios']:,} simulated futures", pad=30)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncols=2, fontsize=9, handlelength=1.2, columnspacing=1.6)
    path = out / "cost.png"
    save(fig, path)
    return path


def anatomy(src: Path, out: Path) -> Path | None:
    file = src / "anatomy.json"
    if not file.is_file():
        return None
    a = json.loads(file.read_text())
    c = a["curve"]
    x = np.array(c["funded"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), sharey=True)

    # Left: the response to the funded ratio alone, at three levels of the yield (one hue, light to dark = low to high yield).
    ax = axes[0]
    ax.plot(x, [w[0] for w in c["glide"]], color=ORANGE, lw=2, label="Glide path (looks at the funded ratio only)")
    names = sorted(c["learned"], key=lambda k: float(k.rstrip("%")))
    shades = ["#9ec5f4", "#5e9be4", "#1c5cab"][:len(names)] if len(names) <= 3 else [BLUE] * len(names)
    ends = []
    for name, shade in zip(names, shades):
        y = [w[0] for w in c["learned"][name]]
        ax.plot(x, y, color=shade, lw=2, label=f"Learned policy, yield at {name}")
        ends.append((y[-1], f"yield {name}"))
    ends.append((c["glide"][-1][0], "glide path"))
    last = None
    for yv, text in sorted(ends, reverse=True):                 # direct labels at the right-hand ends, kept apart
        pos = yv if last is None or last - yv >= 0.055 else last - 0.055
        ax.text(x[-1] + 0.012, pos, text, va="center", fontsize=8, color=INK_2, clip_on=False)
        last = pos
    ax.set_title("Share in stocks as the funded ratio changes")
    ax.set_xlabel("Funded ratio (everything else held at a middling value)")
    ax.xaxis.set_major_formatter(pct)
    ax.yaxis.set_major_formatter(pct)
    ax.set_ylim(0, 1)
    ax.set_xlim(x[0], x[-1])
    ax.set_xticks([0.6, 0.8, 1.0, 1.2, 1.4])
    ax.set_ylabel("Share of the fund")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=2, fontsize=8.5, handlelength=1.4)

    # Right: what it held in the test futures at each level of the yield, for plans a little short of fully funded.
    ax = axes[1]
    b = a["by_yield"]
    keep = [i for i, w in enumerate(b["learned"]) if w is not None]
    xs = np.arange(len(keep))
    stocks = np.array([b["learned"][i][0] for i in keep])
    bonds = np.array([b["learned"][i][1] for i in keep])
    match = np.array([b["learned"][i][2] for i in keep])
    width = 0.62
    ax.bar(xs, stocks, width, color=BLUE, label="Stocks", edgecolor=SURFACE, linewidth=1.5)
    ax.bar(xs, bonds, width, bottom=stocks, color=CATEGORICAL[3], label="10-year bonds", edgecolor=SURFACE, linewidth=1.5)
    ax.bar(xs, match, width, bottom=stocks + bonds, color=CATEGORICAL[2], label="Matching bonds", edgecolor=SURFACE, linewidth=1.5)
    for xi, sv in zip(xs, stocks):
        if sv >= 0.08:
            ax.text(xi, sv / 2, f"{100 * sv:.0f}%", ha="center", va="center", fontsize=9, color="#ffffff")
    glide = float(np.mean([b["glide"][i][0] for i in keep]))
    ax.axhline(glide, color=ORANGE, lw=2)
    ax.text(xs[-1] + width / 2 + 0.08, glide, f"glide path\n{100 * glide:.0f}% in stocks", va="center", fontsize=8, color=INK_2, clip_on=False)
    ax.set_xticks(xs, [b["buckets"][i].replace("%", "").replace(" to ", "–").replace("under ", "<").replace("over ", ">") for i in keep], fontsize=9)
    lo, hi = b["funded_between"]
    ax.set_title(f"What it held, by the yield (plans {100 * lo:.0f}% to {100 * hi:.0f}% funded)")
    ax.set_xlabel("10-year yield at the time of the decision (%), test futures")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=3, fontsize=8.5, handlelength=1.2)
    fig.subplots_adjust(wspace=0.2)
    path = out / "anatomy.png"
    save(fig, path)
    return path


def render(root: Path) -> list[Path]:
    apply_style()
    src, out = root / "reports" / "pension", root / "reports" / "figures" / "pension"
    out.mkdir(parents=True, exist_ok=True)
    return [p for p in (fans(src, out), cost(src, out), anatomy(src, out)) if p is not None]
