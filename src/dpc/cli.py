"""Command-line entry point: ``python -m dpc.cli <stage>``.

Stages: setup, data, signals, baselines, risk, e2e, evaluate, all.
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deep Portfolio Constructor pipeline")
    parser.add_argument("stage", choices=["setup", "data", "signals", "baselines", "risk", "e2e", "evaluate", "all"])
    parser.add_argument("--config", default=None, help="path to config.yaml (default: search upward)")
    parser.add_argument("--refresh", action="store_true", help="re-download data instead of using the cache")
    args = parser.parse_args(argv)
    cfg = _prepare(args.config)
    stages = ["data", "signals", "baselines", "risk", "e2e", "evaluate"] if args.stage == "all" else [args.stage]

    for stage in stages:
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
        elif stage == "e2e":
            from dpc.deep.e2e import run_stage5

            run_stage5(cfg)
        elif stage == "evaluate":
            from dpc.evaluation import run_stage6

            run_stage6(cfg)
        write_run_manifest(run_manifest(cfg.project.seed, cfg.root), cfg.paths.reports / f"run_manifest_{stage}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
