"""Reproducibility helpers: seeding, deterministic torch, and run manifests.

Why this matters in finance research
------------------------------------
A backtest number you cannot reproduce is not evidence. Neural networks add
randomness in weight initialisation, data shuffling and some GPU/CPU kernels.
We remove it so that (a) a reviewer gets the same Sharpe ratio you report and
(b) when a result changes, it changed because the *method* changed.

Good practice used here: rather than relying only on global random state, code
that needs randomness should ask for an explicit generator via ``make_rng`` /
``make_torch_generator``. Explicit generators are easier to reason about and
are not disturbed by unrelated library calls that consume global randomness.
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Packages whose versions are recorded in every run manifest.
TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "torch",
    "cvxpy",
    "cvxpylayers",
    "diffcp",
    "scikit-learn",
    "statsmodels",
    "yfinance",
    "matplotlib",
    "plotly",
    "streamlit",
)


def set_global_seed(seed: int, deterministic_torch: bool = True, torch_num_threads: int | None = None) -> None:
    """Seed python ``random``, numpy's legacy global RNG and torch.

    Parameters
    ----------
    seed:
        Master seed from ``config.yaml``.
    deterministic_torch:
        If True, force deterministic torch kernels (raises on ops that have no
        deterministic implementation, which is what we want to know about).
    torch_num_threads:
        Fix torch's intra-op thread count; different thread counts can change
        floating-point summation order and hence the last bits of results.
    """
    # Only affects *child* processes (hash randomisation is fixed at
    # interpreter start-up), but makes spawned workers consistent.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # legacy global RNG, for libraries that still use it
    torch.manual_seed(seed)  # seeds CPU and all CUDA devices

    if torch_num_threads is not None:
        torch.set_num_threads(torch_num_threads)

    if deterministic_torch:
        # Required by cuBLAS for deterministic matmuls on GPU; harmless on CPU.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_rng(seed: int, stream: int = 0) -> np.random.Generator:
    """Independent numpy Generator for a named purpose.

    ``stream`` lets different components (e.g. bootstrap vs. train/val split)
    draw from statistically independent streams derived from one master seed,
    so adding randomness in one place never shifts the numbers elsewhere.
    """
    return np.random.default_rng(np.random.SeedSequence([seed, stream]))


def make_torch_generator(seed: int, stream: int = 0) -> torch.Generator:
    """Torch equivalent of :func:`make_rng` (e.g. for DataLoader shuffling)."""
    child_seed = int(np.random.SeedSequence([seed, stream]).generate_state(1, dtype=np.uint64)[0] >> 1)
    g = torch.Generator()
    g.manual_seed(child_seed)
    return g


def _git_commit(cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True, timeout=5
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def package_versions(packages: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str | None]:
    """Installed version of each package, or None if not installed."""
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def run_manifest(seed: int, project_root: Path) -> dict[str, Any]:
    """Everything needed to say exactly *how* a result was produced."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "git_commit": _git_commit(project_root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
    }


def write_run_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
