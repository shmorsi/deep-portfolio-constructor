"""Export the figures the portfolio website shows, each with its source file.

`python -m dpc.cli site` writes reports/site_figures.json. The website copies
numbers only from this committed file, and every figure keeps the path of the
table it was read from, so a reader can trace any number on the site.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from dpc.config import Config
from dpc.strategies import LABELS


def _read(cfg: Config, name: str) -> pd.DataFrame:
    return pd.read_csv(cfg.paths.reports / name, index_col=0)


def _test_count(root: Path) -> int:
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=root, capture_output=True, text=True)
    lines = [ln for ln in out.stdout.splitlines() if "::" in ln]
    return len(lines)


def export(cfg: Config) -> dict:
    perf = _read(cfg, "stage6_performance.csv")
    dec = _read(cfg, "stage6_risk_decomposition.csv")
    ic = _read(cfg, "stage4b_alpha_ic.csv")
    risk = _read(cfg, "stage4_risk_model.csv")
    tests = _read(cfg, "stage6_difference_tests.csv")
    hedged = _read(cfg, "stage6_beta_hedged.csv")
    figs: dict = {"_meta": {"oos_start": cfg.backtest.oos_start, "cost_bps": cfg.backtest.transaction_cost_bps}}

    def put(key: str, value: float, source: str) -> None:
        figs[key] = {"value": float(value), "source": source}

    for sid in ("ls_naive", "ls_classical", "ls_deep_twostage", "ls_deep_alpha", "e2e", "e2e_deep"):
        lab = LABELS[sid]
        if lab in perf.index:
            for col in ("sharpe", "ann_return", "ann_vol", "max_drawdown", "alpha_ann_ff6", "alpha_t_ff6", "beta_mkt"):
                put(f"{sid}.{col}", perf.loc[lab, col], f"reports/stage6_performance.csv [{lab}, {col}]")
        if lab in dec.index:
            for col in ("exante_share_specific", "realised_share_specific"):
                put(f"{sid}.{col}", dec.loc[lab, col], f"reports/stage6_risk_decomposition.csv [{lab}, {col}]")
        if lab in hedged.index:
            for col in ("sharpe_unhedged", "sharpe_beta_hedged", "avg_trailing_beta", "alpha_t_hedged"):
                put(f"{sid}.{col}", hedged.loc[lab, col], f"reports/stage6_beta_hedged.csv [{lab}, {col}]")
    for sig in ("deep_alpha", "combined", "momentum"):
        for col in ("mean_IC", "IC_t_stat_NW", "Q5_minus_Q1_monthly", "Q5_Q1_t_stat_NW", "n_months"):
            put(f"ic.{sig}.{col}", ic.loc[sig, col], f"reports/stage4b_alpha_ic.csv [{sig}, {col}]")
    for model in ("deep", "ledoit_wolf", "sample"):
        for col in ("mean_oos_nll", "gmv_realised_vol_ann", "share_months_best_nll"):
            put(f"risk.{model}.{col}", risk.loc[model, col], f"reports/stage4_risk_model.csv [{model}, {col}]")
    for row in tests.index:
        for col in ("mean_monthly_diff", "t_stat", "n_months"):
            put(f"diff.{row}.{col}", tests.loc[row, col], f"reports/stage6_difference_tests.csv [{row}, {col}]")
    for name in ("stage6_seeds.csv", "stage6_seeds_deep.csv"):
        p = cfg.paths.reports / name
        if p.exists():
            s = pd.read_csv(p, index_col=0)
            tag = "seeds_deep" if "deep" in name else "seeds"
            for idx in s.index:
                put(f"{tag}.{idx}.sharpe", s.loc[idx, "sharpe"], f"reports/{name} [{idx}, sharpe]")
    figs["tests"] = {"value": _test_count(cfg.root), "source": "pytest --collect-only, deep-portfolio-constructor main"}

    checks = json.loads((cfg.paths.reports / "stage6_lookahead_checks.json").read_text())
    put("oracle_sharpe", checks["oracle_sharpe"], "reports/stage6_lookahead_checks.json [oracle_sharpe]")
    put("lookahead_checks_passed", float(all(v for k, v in checks.items() if k != "oracle_sharpe")),
        "reports/stage6_lookahead_checks.json [engine and deep-feature truncation invariance]")
    cost = _read(cfg, "stage6_cost_sensitivity.csv")
    for sid in ("e2e", "e2e_deep", "ls_classical"):
        lab = LABELS[sid]
        if lab in cost.index:
            for col in cost.columns:
                put(f"cost.{sid}.{col}", cost.loc[lab, col], f"reports/stage6_cost_sensitivity.csv [{lab}, {col}]")
    cons = pd.read_csv(cfg.paths.reports / "stage6_constraint_sensitivity.csv")
    for sid in ("e2e", "e2e_deep"):
        sub = cons[cons["strategy"] == LABELS[sid]]
        if len(sub):
            put(f"constraints.{sid}.sharpe_min", sub["sharpe"].min(), f"reports/stage6_constraint_sensitivity.csv [{sid} rows, min sharpe]")
            put(f"constraints.{sid}.sharpe_max", sub["sharpe"].max(), f"reports/stage6_constraint_sensitivity.csv [{sid} rows, max sharpe]")
    ms = cfg.paths.reports / "stage6_model_size.csv"
    if ms.exists():
        m = pd.read_csv(ms, index_col=0)
        for idx in m.index:
            for col in ("n_params", "oos_nll_deep", "gmv_vol_deep", "twostage_sharpe"):
                put(f"model_size.{idx}.{col}", m.loc[idx, col], f"reports/stage6_model_size.csv [{idx}, {col}]")
    q = _read(cfg, "data_quality_summary.csv")
    put("universe.n_stocks", int(q["passed"].sum()), "reports/data_quality_summary.csv [passed]")

    # Parameter counts of the PyTorch models, as configured
    from dpc.deep.alpha_model import AlphaFormer
    from dpc.deep.risk_model import DeepRiskModel

    n_sec = 11
    alpha = AlphaFormer(2, 3 + 6 + 3 + n_sec, cfg.alpha_model)
    risk = DeepRiskModel(6, 2 + n_sec, cfg.risk_model)
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    put("params.alpha_former", count(alpha), "src/dpc/deep/alpha_model.py with config.yaml alpha_model, 3-seed ensemble has 3x")
    put("params.risk_model", count(risk), "src/dpc/deep/risk_model.py with config.yaml risk_model")
    out = cfg.paths.reports / "site_figures.json"
    out.write_text(json.dumps(figs, indent=2) + "\n")
    return figs


def render_site_figures(cfg: Config) -> list[Path]:
    """Dark-surface versions of the key charts, for the website (reports/figures/site/)."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FuncFormatter

    from dpc.data.panel import Panel
    from dpc.plotting import DARK, apply_dark_style, dark, save
    from dpc.signals import forward_returns, information_coefficients, monthly_returns
    from dpc.strategies import COLORS, LABELS, LONG_SHORT, SHORT, load_all_results

    apply_dark_style()
    out_dir = cfg.paths.figures / "site"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    results = load_all_results(cfg.paths.results)
    ls = [s for s in LONG_SHORT if s in results]

    # 1. Where the risk comes from
    dec = pd.read_csv(cfg.paths.reports / "stage6_risk_decomposition.csv", index_col=0)
    rows = [s for s in ls if LABELS[s] in dec.index]
    d = dec.loc[[LABELS[s] for s in rows], ["exante_share_specific", "exante_share_style", "exante_share_sector"]].clip(lower=0)
    d = d.div(d.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(10, 5.625))
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    parts = [("exante_share_specific", "Stock-specific (intended bets)", dark("#2a78d6")),
             ("exante_share_style", "Style factors", dark("#eb6834")),
             ("exante_share_sector", "Sectors", dark("#eda100"))]
    for col, label, color in parts:
        vals = d[col].to_numpy()
        ax.barh(y, vals, left=left, color=color, height=0.6, label=label, edgecolor=DARK["surface"], linewidth=2)
        for yi, l, val in zip(y, left, vals):
            if val >= 9:
                ax.text(l + val / 2, yi, f"{val:.0f}%", ha="center", va="center", color="#ffffff" if col != "exante_share_sector" else "#0b0b0b", fontsize=9)
        left += vals
    ax.set_yticks(y, [SHORT[s] for s in rows])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of predicted portfolio variance (%)")
    ax.set_title("Where the risk comes from, market-neutral strategies")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncols=3, fontsize=9)
    p = out_dir / "dpc-risk.png"
    save(fig, p)
    written.append(p)

    # 2. Cumulative net returns of the market-neutral strategies
    fig, ax = plt.subplots(figsize=(10, 5.625))
    for s in ls:
        w = (1 + results[s].daily_net).cumprod()
        ax.plot(w.index, w.values, color=dark(COLORS[s]), lw=2.4 if s == "e2e" else 1.5, label=SHORT[s])
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"${val:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda val, _: f"${val:g}" if f"{val:g}".lstrip("0.")[:1] in ("2", "5") else ""))
    ax.axhline(1, color=DARK["baseline"], lw=1)
    ax.set_title("Growth of $1, net of 10 bps costs, out of sample from 2008")
    ax.legend(loc="upper left", fontsize=8.5, ncols=2)
    p = out_dir / "dpc-cumulative.png"
    save(fig, p)
    written.append(p)

    # 3. Market beta: estimated at each rebalance (held inside the band) vs realised (the leak)
    perf = pd.read_csv(cfg.paths.reports / "stage6_performance.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    lim = cfg.optimizer.factor_limit
    ax.axhspan(-lim, lim, color=DARK["grid"], alpha=0.9, lw=0)
    ax.axhline(0, color=DARK["baseline"], lw=1)
    for s in ("ls_classical", "e2e"):
        f = cfg.paths.results / f"{s}__risk_decomp.parquet"
        if f.exists():
            ex = pd.read_parquet(f)["exp_style:Mkt-RF"]
            ax.plot(ex.index, ex.values, color=dark(COLORS[s]), lw=1.8 if s == "e2e" else 1.3, label=f"{SHORT[s]}, estimated")
    realised = float(perf.loc[LABELS["e2e"], "beta_mkt"])
    ax.axhline(realised, color=dark(COLORS["e2e"]), lw=1.6, ls=(0, (5, 4)), label=f"{SHORT['e2e']}, realised: {realised:.2f}")
    ax.set_ylim(-0.12, max(0.25, realised + 0.05))
    ax.set_title(f"Market beta: estimated at each rebalance (inside the ±{lim:g} band) vs realised")
    ax.set_ylabel("Portfolio beta")
    ax.legend(loc="upper left", fontsize=8.5, ncols=3)
    p = out_dir / "dpc-beta.png"
    save(fig, p)
    written.append(p)

    # 4. Cumulative IC: AlphaFormer vs classical signals
    panel = Panel.load(cfg.paths.data_processed)
    fwd = forward_returns(monthly_returns(panel.returns, panel.month_ends))
    deep_sig = pd.read_parquet(cfg.paths.data_processed / "signal_deep_alpha.parquet")
    oos = deep_sig.index[deep_sig.notna().any(axis=1)]
    sigs = {"AlphaFormer (deep)": (deep_sig, dark("#2a78d6"), 2.4)}
    for k, c in (("combined", "#1baf7a"), ("momentum", "#eb6834"), ("low_vol", "#eda100")):
        sigs[k.replace("_", " ")] = (pd.read_parquet(cfg.paths.data_processed / f"signal_{k}.parquet"), dark(c), 1.4)
    fig, ax = plt.subplots(figsize=(10, 5.0))
    for name, (sig, color, lw) in sigs.items():
        ic = information_coefficients(sig.reindex(oos), fwd.reindex(oos)).cumsum()
        ax.plot(ic.index, ic.values, color=color, lw=lw, label=name)
    ax.axhline(0, color=DARK["baseline"], lw=1)
    ax.set_title("Out-of-sample cumulative rank IC: deep views vs classical signals")
    ax.set_ylabel("Sum of monthly rank IC")
    ax.legend(loc="upper left", fontsize=8.5)
    p = out_dir / "dpc-ic.png"
    save(fig, p)
    written.append(p)
    return written
