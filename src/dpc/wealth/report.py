"""Figures for the README and research note, and the export the website reads.

* reports/figures/wealth/*.png: growth of $1, the network's tilts by decade,
  rolling 10-year growth against the market, and drawdowns.
* reports/site_wealth.json: every figure and series the website shows, each
  traced to the file it was computed from. The website copies it verbatim.
"""

from __future__ import annotations

import json
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..config import Config  # noqa: E402
from ..logging_utils import get_logger  # noqa: E402
from .names import ABLATIONS, COLORS, LABELS, MAIN, REFERENCE  # noqa: E402
from .model import DeepGrowthNet  # noqa: E402
from .pipeline import load, load_result, results_dir  # noqa: E402

log = get_logger(__name__)


def _bp(row: np.ndarray) -> list[int]:
    """Weights in whole basis points that add up to exactly 10,000 (largest-remainder rounding)."""
    exact = 10000 * row / row.sum()
    floors = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - floors))[:10000 - floors.sum()]:
        floors[i] += 1
    return floors.tolist()


def _params(cfg: Config, panel) -> int:
    """Trainable parameters of one Deep Growth Network, as configured."""
    from .features import ASSET_FEATURES, CONTEXT_FEATURES

    m, w = cfg.wealth.model, cfg.wealth
    net = DeepGrowthNet(len(ASSET_FEATURES), len(CONTEXT_FEATURES), m.d_model, m.n_heads, m.n_layers, m.dropout, w.max_tilt, m.init_trade)
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def _test_count(cfg: Config) -> int:
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=cfg.root, capture_output=True, text=True)
    return len([ln for ln in out.stdout.splitlines() if "::" in ln])


def figures(cfg: Config, results: dict[str, pd.DataFrame], panel) -> list[str]:
    out = cfg.paths.figures / "wealth"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    written = []

    fig, ax = plt.subplots(figsize=(10, 5.2))
    for name in MAIN:
        if name not in results:
            continue
        w = (1 + results[name]["net"]).cumprod()
        ax.plot(w.index, w.values, color=COLORS[name], lw=2.4 if name == "deep" else 1.2,
                alpha=1 if name in ("deep", "market") else 0.75, label=f"{LABELS[name]}: ${w.iloc[-1]:,.0f}")
    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.set_title("Growth of $1 invested in January 1960, long-only, net of 10 bps costs", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.25)
    p = out / "growth.png"
    fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig); written.append(str(p))

    if "deep" in results:
        deep = results["deep"]
        w = deep[[c for c in deep.columns if c.startswith("w:")]]
        start = panel.index_of(cfg.wealth.oos_start)
        m = pd.DataFrame(panel.market_weights()[start - 1:panel.n_months - 1], index=w.index, columns=w.columns)
        tilt = np.log(w / m).groupby((w.index.year // 5) * 5).mean().T
        tilt.index = [c[2:] for c in tilt.index]
        fig, ax = plt.subplots(figsize=(10, 7))
        lim = float(np.log(cfg.wealth.max_tilt))
        im = ax.imshow(tilt.values, aspect="auto", cmap="RdBu", vmin=-lim, vmax=lim)
        ax.set_yticks(range(len(tilt.index)), tilt.index, fontsize=7.5)
        ax.set_xticks(range(len(tilt.columns)), [f"{c}" for c in tilt.columns], rotation=90, fontsize=8)
        ax.set_title("Deep Growth Network: tilt against market weights by industry, 5-year averages (blue = overweight)", loc="left", fontsize=10)
        cb = fig.colorbar(im, ax=ax, fraction=0.025)
        cb.set_ticks([-lim, 0, lim], labels=[f"1/{cfg.wealth.max_tilt:g}x", "market", f"{cfg.wealth.max_tilt:g}x"])
        p = out / "tilts.png"
        fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig); written.append(str(p))

    fig, ax = plt.subplots(figsize=(10, 4.2))
    mk = np.log1p(results["market"]["net"]).rolling(120).sum() / 10
    for name in ("deep", "momentum_tilt", "equal_weight"):
        if name in results:
            r = np.log1p(results[name]["net"]).rolling(120).sum() / 10 - mk
            ax.plot(r.index, 100 * r.values, color=COLORS[name], lw=2 if name == "deep" else 1.2, label=LABELS[name])
    ax.axhline(0, color="#111", lw=0.8)
    ax.set_ylabel("Growth above the market, % a year")
    ax.set_title("Rolling 10-year growth against the market", loc="left")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(alpha=0.25)
    p = out / "rolling.png"
    fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig); written.append(str(p))
    return written


def export_site(cfg: Config) -> dict:
    """reports/site_wealth.json: the figures and series the website shows."""
    panel, _ = load(cfg)
    summary = json.loads((cfg.paths.reports / "wealth" / "summary.json").read_text())
    available = {p.stem for p in results_dir(cfg).glob("*.parquet")}
    results = {n: load_result(cfg, n) for n in MAIN + ABLATIONS + REFERENCE if n in available}
    figures(cfg, results, panel)
    commit = subprocess.run(["git", "-C", str(cfg.root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    mkt = results["market"]
    dates = [f"{d:%Y-%m}" for d in mkt.index]
    series = {n: [1.0] + [round(float(v), 4) for v in (1 + df["net"]).cumprod()] for n, df in results.items()}
    deep = results.get("deep")
    holdings = None
    if deep is not None:
        wcols = [c for c in deep.columns if c.startswith("w:")]
        start = panel.index_of(cfg.wealth.oos_start)
        mw = panel.market_weights()[start - 1:panel.n_months - 1]
        holdings = {"industries": [c[2:] for c in wcols], "dates": dates,
                    "deepBp": [_bp(r) for r in deep[wcols].to_numpy()], "marketBp": [_bp(r) for r in mw],
                    "kappa": [round(float(k), 4) for k in deep["kappa"]] if "kappa" in deep else None}
    site = {
        "source": f"deep-portfolio-constructor {commit}: reports/results/wealth/*.parquet and reports/wealth/summary.json",
        "commit": commit,
        "oos": summary["oos"], "cost_bps": summary["cost_bps"],
        "labels": {n: LABELS[n] for n in results},
        "start": f"{(mkt.index[0] - pd.offsets.MonthEnd(1)):%Y-%m}",
        "dates": dates,
        "growth": series,
        "performance": summary["performance"],
        "comparisons": summary["comparisons"],
        "decades": summary["decades"],
        "costs": summary["costs"],
        "seeds": summary["seeds"],
        "oracle_terminal_wealth": summary["oracle_terminal_wealth"],
        "trade_fraction": summary.get("deep_trade_fraction"),
        "holdings": holdings,
        "params": _params(cfg, panel),
        "tests": _test_count(cfg),
        "data_check": json.loads((cfg.paths.reports / "wealth" / "data_summary.json").read_text())["check_cap_weighted_vs_french_market"],
    }
    (cfg.paths.reports / "site_wealth.json").write_text(json.dumps(site, separators=(",", ":")) + "\n")
    (cfg.paths.reports / "wealth" / "results.md").write_text(results_markdown(site))
    # Each variant's training history (validation growth and best step per refit and seed), kept with the results:
    # the weights themselves are re-creatable and stay out of git.
    hist_dir = cfg.paths.reports / "wealth" / "training"
    hist_dir.mkdir(parents=True, exist_ok=True)
    for meta in sorted((cfg.paths.models / "wealth").glob("*/training.json")):
        (hist_dir / f"{meta.parent.name}.json").write_text(meta.read_text())
    log.info("Site export written: %s", cfg.paths.reports / "site_wealth.json")
    return site


def _pct(x: float | None, digits: int = 2, sign: bool = False) -> str:
    if x is None:
        return "n/a"
    return f"{100 * x:+.{digits}f}%" if sign else f"{100 * x:.{digits}f}%"


def results_markdown(site: dict) -> str:
    """reports/wealth/results.md: every table of the study, generated from the same export as the website."""
    perf, comps, labels = site["performance"], site["comparisons"], site["labels"]
    order = [n for n in MAIN + ABLATIONS + REFERENCE if n in perf]
    out = [f"# Wealth study: results\n",
           f"Generated by `python -m dpc.cli wealth-site` from `reports/results/wealth/*.parquet` (commit {site['commit']}). "
           f"Out of sample {site['oos']['start']} to {site['oos']['end']} ({site['oos']['months']} months), long-only, fully invested, "
           f"{site['cost_bps']:g} bps per unit traded. Do not edit by hand.\n"]
    if "deep - market" in comps:
        c = comps["deep - market"]
        out.append(f"**Primary comparison (pre-registered):** Deep Growth Network minus the market, {_pct(c['diff_log_growth'], sign=True)} a year "
                   f"in log growth, Newey-West t {c['t_nw']:.2f}, 95% block-bootstrap interval {_pct(c['ci_low'], sign=True)} to "
                   f"{_pct(c['ci_high'], sign=True)}: **{c['verdict']}**.\n")
    out.append("## Performance\n")
    out.append("| Strategy | $1 became | CAGR | Volatility | Sharpe | Max drawdown | Worst year | Traded / year | Active share | FF3 alpha (t) | Beat market, rolling 10y |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for n in order:
        p = perf[n]
        out.append(f"| {labels[n]} | ${p['terminal_wealth']:,.0f} | {_pct(p['cagr'])} | {_pct(p['volatility'], 1)} | {p['sharpe']:.2f} | "
                   f"{_pct(p['max_drawdown'], 0)} | {_pct(p['worst_year'], 0)} | {_pct(p['turnover_per_year'], 0)} | {_pct(p.get('active_share'), 0)} | "
                   f"{_pct(p['alpha_annual'], sign=True)} ({p['alpha_t']:.2f}) | {_pct(p['beat_market_rolling'], 0)} |")
    out.append("\n## Comparisons (difference in annualised net log growth)\n")
    out.append("| Comparison | Difference / year | Newey-West t | 95% bootstrap interval | Verdict |")
    out.append("|---|---:|---:|---|---|")
    for key, c in comps.items():
        a, b = key.split(" - ")
        out.append(f"| {labels.get(a, a)} minus {labels.get(b, b) if b != 'market' else 'the market'} | {_pct(c['diff_log_growth'], sign=True)} | "
                   f"{c['t_nw']:.2f} | {_pct(c['ci_low'], sign=True)} to {_pct(c['ci_high'], sign=True)} | {c['verdict']} |")
    cols = sorted(site["decades"])
    out.append("\n## Compound growth per year by decade\n")
    out.append("| Strategy | " + " | ".join(cols) + " |")
    out.append("|---|" + "---:|" * len(cols))
    for n in order:
        out.append(f"| {labels[n]} | " + " | ".join(_pct(site["decades"][c].get(n), 1) for c in cols) + " |")
    out.append("\n## Costs: the same decisions re-priced\n")
    out.append("| Strategy at cost | $1 became | CAGR | Growth vs the market at the same cost |")
    out.append("|---|---:|---:|---:|")
    for key, c in site["costs"].items():
        n, bps = key.split("@")
        out.append(f"| {labels.get(n, n)} at {bps} | ${c['terminal_wealth']:,.0f} | {_pct(c['cagr'])} | {_pct(c['vs_market_log_growth'], sign=True)} |")
    if site["seeds"]:
        out.append("\n## Each seed's network on its own\n")
        out.append("| Seed | $1 became | CAGR | Sharpe | Max drawdown | Traded / year |")
        out.append("|---|---:|---:|---:|---:|---:|")
        for k in sorted(site["seeds"]):
            p = site["seeds"][k]
            out.append(f"| {k.removeprefix('deep_seed')} | ${p['terminal_wealth']:,.0f} | {_pct(p['cagr'])} | {p['sharpe']:.2f} | "
                       f"{_pct(p['max_drawdown'], 0)} | {_pct(p['turnover_per_year'], 0)} |")
    out.append("\n## Checks\n")
    if site.get("trade_fraction"):
        tf = site["trade_fraction"]
        out.append(f"* Trade fraction (share of the gap to the target traded each month), Deep Growth Network: median {_pct(tf['median'], 0)}, "
                   f"mean {_pct(tf['mean'], 0)}, range {_pct(tf['min'], 0)} to {_pct(tf['max'], 0)}. Training starts it at 25%.")
    out.append(f"* Calibration: a deliberately cheating policy that tilts toward next month's best industries turns $1 into "
               f"${site['oracle_terminal_wealth']:,.0f}.")
    dc = site["data_check"]
    out.append(f"* Data: the cap-weighted portfolio of the 30 industries tracks French's market return with correlation {dc['correlation']:.4f} "
               f"(mean absolute difference {100 * dc['mean_abs_diff_monthly']:.3f}% a month).")
    out.append(f"* {site['tests']} automated tests; one Deep Growth Network has {site['params']:,} parameters.")
    return "\n".join(out) + "\n"
