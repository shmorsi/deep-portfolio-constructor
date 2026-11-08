"""Command-line entry point: ``python -m dpc.cli <stage>``.

Stages: setup, quickstart, data, signals, baselines, risk, alpha, e2e, robustness, evaluate, all.
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

    Prints the headline table from the committed evaluation files and opens the
    dashboard, which reads only committed results. Reproducing every number
    from raw data is `python -m dpc.cli all` (about 2 hours on a laptop CPU).
    """
    import subprocess

    import pandas as pd

    perf_path = cfg.paths.reports / "stage6_performance.csv"
    if not perf_path.exists():
        print("No committed results found. Run `python -m dpc.cli all` first.")
        return 1
    perf = pd.read_csv(perf_path, index_col=0)
    cols = ["sharpe", "ann_return", "ann_vol", "max_drawdown", "alpha_t_ff6"]
    pd.set_option("display.width", 140)
    print("\nOut-of-sample, Jan 2008 onward, monthly rebalancing, net of costs:\n")
    print(perf[cols].round(3).to_string())
    print("\nFirst-principles walkthrough: notebooks/01_first_principles.ipynb")
    print("Research note:                reports/research_note.md")
    if launch:
        app = cfg.root / "dashboard" / "app.py"
        print(f"\nOpening the dashboard ({app.relative_to(cfg.root)}); press Ctrl+C to stop.\n")
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)], cwd=cfg.root)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deep Portfolio Constructor pipeline")
    parser.add_argument("stage", choices=["setup", "quickstart", "data", "signals", "baselines", "risk", "alpha", "e2e", "robustness", "evaluate", "site", "all",
                                          "wealth", "wealth-data", "wealth-baselines", "wealth-train", "wealth-evaluate", "wealth-site"])
    parser.add_argument("--variants", default="deep", help="wealth-train: comma-separated variants, or 'all' (deep + ablations)")
    parser.add_argument("--config", default=None, help="path to config.yaml (default: search upward)")
    parser.add_argument("--refresh", action="store_true", help="re-download data instead of using the cache")
    parser.add_argument("--no-launch", action="store_true", help="quickstart: print results only, do not open the dashboard")
    args = parser.parse_args(argv)
    cfg = _prepare(args.config)
    if args.stage == "all":
        stages = ["data", "signals", "baselines", "risk", "alpha", "e2e", "robustness", "evaluate"]
    elif args.stage == "wealth":
        stages = ["wealth-data", "wealth-baselines", "wealth-train", "wealth-evaluate", "wealth-site"]
    else:
        stages = [args.stage]

    for stage in stages:
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

                names = list(VARIANTS) if args.variants == "all" else args.variants.split(",")
                pipeline.stage_train(cfg, names)
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
