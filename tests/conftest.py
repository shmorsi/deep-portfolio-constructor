"""Shared fixtures: the real config and (if built) the real data panel."""

from __future__ import annotations

from pathlib import Path

import pytest

from dpc.config import Config, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg() -> Config:
    return load_config(PROJECT_ROOT / "config.yaml")


@pytest.fixture(scope="session")
def panel(cfg: Config):
    """The processed panel; tests using it are skipped until Stage 1 has run."""
    from dpc.data.panel import Panel

    try:
        return Panel.load(cfg.paths.data_processed)
    except FileNotFoundError:
        pytest.skip("processed panel not built (run `python -m dpc.cli data`)")
