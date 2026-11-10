"""Command-line entry point: ``python -m dpc.cli <stage>``.

The long-only wealth study: quickstart, wealth (every stage below, in order),
wealth-data, wealth-baselines, wealth-train, wealth-reference, wealth-benchmarks, wealth-schedules,
wealth-evaluate, wealth-site.
The archived long-short study (docs/longshort.md): setup, data, signals, baselines, risk,
alpha, e2e, robustness, evaluate, site, all.
"""

from __future__ import annotations

import argparse
import sys

from dpc.config import Config, ensure_directories, load_config
from dpc.logging_utils import setup_logging
from dpc.reproducibility import run_manifest, set_global_seed, write_run_manifest


def _prepare(config_path: str | None) -> Config:
    cfg = load_config(config_path)
    setup_logging(cfg.logging.level)
    set_global_seed(cfg.project.seed, cfg.project.deterministic_torch, cfg.project.torch_num_threads)
    ensure_directories(cfg)
    return cfg


def quickstart(cfg: Config, launch: bool = True) -> int:
    """See the results in about a minute: no download, no training, no GPU.

    Prints the long-only wealth study's headline table from the committed
    evaluation and opens the dashboard, which reads only committed results.
    Reproducing every number from raw data is `python -m dpc.cli wealth`.
    """
    import json
    import subprocess

    import pandas as pd

    from dpc.wealth.names import LABELS, MAIN

    summary_path = cfg.paths.reports / "wealth" / "summary.json"
    if not summary_path.exists():
        print("No committed results found. Run `python -m dpc.cli wealth` first.")
        return 1
    s = json.loads(summary_path.read_text())
    rows = [{"strategy": LABELS[n], "$1 became": f"${p['terminal_wealth']:,.0f}", "CAGR": f"{100 * p['cagr']:.2f}%",
             "Sharpe": f"{p['sharpe']:.2f}", "worst fall": f"{100 * p['max_drawdown']:.0f}%", "traded a year": f"{100 * p['turnover_per_year']:.0f}%"}
            for n in MAIN if (p := s["performance"].get(n))]
    pd.set_option("display.width", 140)
    print(f"\n$1 invested in January 1960, out of sample to {s['oos']['end']}: long-only, fully invested, {s['cost_bps']:g} bps per trade.\n")
    print(pd.DataFrame(rows).to_string(index=False))
    if (c := s["comparisons"].get("deep - market")):
        print(f"\nPre-registered test, Deep Growth Network minus the market: {100 * c['diff_log_growth']:+.2f}% a year, "
              f"Newey-West t {c['t_nw']:.2f}: {c['verdict']}.")
    print("\nResearch note: reports/wealth/research_note.md")
    print("Every table:   reports/wealth/results.md")
    print("Protocol:      docs/wealth_protocol.md")
    if launch:
        app = cfg.root / "dashboard" / "app.py"
        print(f"\nOpening the dashboard ({app.relative_to(cfg.root)}); press Ctrl+C to stop.\n")
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)], cwd=cfg.root)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deep Portfolio Constructor pipeline")
    parser.add_argument("stage", choices=["setup", "quickstart", "data", "signals", "baselines", "risk", "alpha", "e2e", "robustness", "evaluate", "site", "all",
                                          "wealth", "wealth-data", "wealth-baselines", "wealth-train", "wealth-reference", "wealth-benchmarks", "wealth-schedules", "wealth-evaluate", "wealth-site",
                                          "pension", "pension-run", "pension-site"])
    parser.add_argument("--variants", default=None,
                        help="wealth-train: comma-separated variants, or 'all' (deep + ablations); default deep, or all within `wealth`")
    parser.add_argument("--workers", type=int, default=None, help="wealth-train: parallel seed processes (default: config)")
    parser.add_argument("--config", default=None, help="path to config.yaml (default: search upward)")
    parser.add_argument("--refresh", action="store_true", help="re-download data instead of using the cache")
    parser.add_argument("--no-launch", action="store_true", help="quickstart: print results only, do not open the dashboard")
    args = parser.parse_args(argv)
    cfg = _prepare(args.config)
    if args.stage == "all":
        stages = ["data", "signals", "baselines", "risk", "alpha", "e2e", "robustness", "evaluate"]
    elif args.stage == "wealth":
        stages = ["wealth-data", "wealth-baselines", "wealth-train", "wealth-benchmarks", "wealth-schedules", "wealth-evaluate", "wealth-site"]
    elif args.stage == "pension":
        stages = ["pension-run", "pension-site"]
    else:
        stages = [args.stage]

    for stage in stages:
        if stage == "pension-run":
            from dpc.pension.run import study

            study(cfg.root, refresh=args.refresh)
            continue
        if stage == "pension-site":
            from dpc.pension.site import export

            export(cfg.root)
            print("wrote reports/pension/site_pension.json")
            continue
        if stage == "quickstart":
            return quickstart(cfg, launch=not args.no_launch)
        if stage == "setup":
            from dpc.setup_check import main as setup_main

            return setup_main(cfg)
        if stage == "data":
            from dpc.data.panel import run_stage1

            run_stage1(cfg, refresh=args.refresh)
        elif stage == "signals":
            from dpc.signals import run_stage2

            run_stage2(cfg)
        elif stage == "baselines":
            from dpc.baselines import run_stage3

            run_stage3(cfg)
        elif stage == "risk":
            from dpc.deep.train_risk import run_stage4

            run_stage4(cfg)
        elif stage == "alpha":
            from dpc.deep.train_alpha import run_stage4b

            run_stage4b(cfg)
        elif stage == "e2e":
            from dpc.deep.e2e import run_stage5

            run_stage5(cfg)
            run_stage5(cfg, tag="deep", views=True)
        elif stage == "robustness":
            from dpc.robustness import run_robustness

            run_robustness(cfg)
        elif stage == "site":
            from dpc.site_export import export, render_site_figures

            export(cfg)
            render_site_figures(cfg)
        elif stage == "evaluate":
            from dpc.evaluation import run_stage6

            run_stage6(cfg)
        elif stage.startswith("wealth-"):
            from dpc.wealth import pipeline

            if stage == "wealth-data":
                pipeline.stage_data(cfg, refresh=args.refresh)
            elif stage == "wealth-baselines":
                pipeline.stage_baselines(cfg)
            elif stage == "wealth-train":
                from dpc.wealth.train import VARIANTS

                chosen = args.variants or ("all" if args.stage == "wealth" else "deep")
                names = list(VARIANTS) if chosen == "all" else chosen.split(",")
                pipeline.stage_train(cfg, names, workers=args.workers)
            elif stage == "wealth-reference":
                pipeline.stage_reference(cfg)
            elif stage == "wealth-benchmarks":
                pipeline.stage_benchmarks(cfg, refresh=args.refresh)
            elif stage == "wealth-schedules":
                pipeline.stage_schedules(cfg, refresh=args.refresh)
            elif stage == "wealth-evaluate":
                from dpc.wealth.evaluate import run_evaluation

                run_evaluation(cfg)
            elif stage == "wealth-site":
                from dpc.wealth.report import export_site

                export_site(cfg)
        write_run_manifest(run_manifest(cfg.project.seed, cfg.root), cfg.paths.reports / f"run_manifest_{stage}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
