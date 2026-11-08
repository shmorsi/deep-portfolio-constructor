"""Strict, typed loading of ``config.yaml``.

Design choices
--------------
* Each YAML section maps to a frozen dataclass, so settings are typed,
  autocompleted in an IDE and cannot be mutated by accident mid-run.
* Loading is strict: a missing key or an unknown key raises ``ConfigError``.
  A silent default is a hidden magic number, which this project forbids.
* Relative paths are resolved against the folder holding ``config.yaml``, so
  scripts behave the same whatever directory they are launched from.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar, Union

import yaml

T = TypeVar("T")

DEFAULT_CONFIG_NAME = "config.yaml"
VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


class ConfigError(ValueError):
    """Raised when ``config.yaml`` is missing, malformed or inconsistent."""


# --------------------------------------------------------------------------- #
# Section schemas
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProjectConfig:
    name: str
    seed: int
    deterministic_torch: bool
    torch_num_threads: int
    device: str


@dataclass(frozen=True)
class PathsConfig:
    """Absolute paths (resolved at load time)."""

    data_raw: Path
    data_processed: Path
    models: Path
    reports: Path
    figures: Path
    notebooks: Path
    results: Path
    universe_file: Path

    def all(self) -> list[Path]:
        """All directory paths (files excluded)."""
        return [getattr(self, f.name) for f in fields(self) if not f.name.endswith("_file")]


@dataclass(frozen=True)
class LoggingConfig:
    level: str


@dataclass(frozen=True)
class QualityConfig:
    outlier_abs_daily_return: float
    split_day_abs_return_max: float
    max_missing_frac: float
    max_stale_run_days: int


@dataclass(frozen=True)
class UniverseConfig:
    min_history_days: int
    max_missing_in_window: int


@dataclass(frozen=True)
class DataConfig:
    start_date: str
    end_date: str | None
    benchmark_ticker: str
    ff_base_url: str
    ff_five_factor_file: str
    ff_momentum_file: str
    membership_file: str | None
    quality: QualityConfig
    universe: UniverseConfig


@dataclass(frozen=True)
class SignalWeights:
    momentum: float
    reversal: float
    low_vol: float


@dataclass(frozen=True)
class SignalsConfig:
    momentum_lookback_months: int
    momentum_skip_months: int
    reversal_lookback_months: int
    low_vol_lookback_days: int
    winsorize_z: float
    weights: SignalWeights
    newey_west_lags: int


@dataclass(frozen=True)
class ExposuresConfig:
    beta_window_days: int
    beta_min_obs: int


@dataclass(frozen=True)
class BacktestConfig:
    oos_start: str
    transaction_cost_bps: float
    cov_window_days: int
    annualization_days: int
    min_weight_print: float


@dataclass(frozen=True)
class AlphaConfig:
    assumed_ic: float


@dataclass(frozen=True)
class BaselinesConfig:
    long_only_max_weight: float
    hrp_linkage: str


@dataclass(frozen=True)
class OptimizerConfig:
    gross_leverage: float
    max_weight: float
    sector_limit: float
    factor_limit: float
    neutral_factors: list[str]
    turnover_limit: float
    risk_aversion: float
    turnover_penalty_mult: float
    solver_eps: float


@dataclass(frozen=True)
class RiskModelConfig:
    architecture: str
    seq_len_days: int
    patch_len_days: int
    horizon_days: int
    n_factors: int
    d_model: int
    n_heads: int
    temporal_layers: int
    cross_sectional_layers: int
    dropout: float
    learning_rate: float
    weight_decay: float
    max_epochs: int
    batch_size: int
    sample_stride_days: int
    val_fraction: float
    early_stopping_patience: int
    refit_every_months: int
    warm_start: bool
    min_idio_var: float
    grad_clip: float


@dataclass(frozen=True)
class AlphaModelConfig:
    seq_len_months: int
    d_model: int
    n_heads: int
    temporal_layers: int
    cross_sectional_layers: int
    dropout: float
    learning_rate: float
    weight_decay: float
    max_epochs: int
    batch_size: int
    val_months: int
    early_stopping_patience: int
    refit_every_months: int
    warm_start: bool
    ensemble_size: int
    feature_clip: float
    target_clip: float
    grad_clip: float


@dataclass(frozen=True)
class E2EConfig:
    learning_rate: float
    pretrained_lr_mult: float
    weight_decay: float
    max_epochs: int
    batch_size: int
    val_months: int
    early_stopping_patience: int
    loss_risk_aversion: float
    alpha_hidden: int
    alpha_residual_scale: float
    finetune_risk_model: bool
    grad_clip: float


@dataclass(frozen=True)
class EvaluationConfig:
    crisis_years: list[int]
    sensitivity_cost_bps: list[float]
    sensitivity_sector_limits: list[float]
    sensitivity_max_weights: list[float]
    sensitivity_d_model: list[int]
    seeds: list[int]
    risk_model_window_days: int
    hedge_beta_window_days: int


@dataclass(frozen=True)
class DashboardConfig:
    default_strategy: str


@dataclass(frozen=True)
class Config:
    """Top-level configuration object passed through the whole pipeline."""

    root: Path
    project: ProjectConfig
    paths: PathsConfig
    logging: LoggingConfig
    data: DataConfig
    signals: SignalsConfig
    exposures: ExposuresConfig
    backtest: BacktestConfig
    alpha: AlphaConfig
    baselines: BaselinesConfig
    optimizer: OptimizerConfig
    risk_model: RiskModelConfig
    alpha_model: AlphaModelConfig
    e2e: E2EConfig
    evaluation: EvaluationConfig
    dashboard: DashboardConfig

    def replace(self, **sections: Any) -> "Config":
        """Copy with whole sections swapped (used by sensitivity tests)."""
        return dataclasses.replace(self, **sections)


# --------------------------------------------------------------------------- #
# Generic strict builder
# --------------------------------------------------------------------------- #
def _check_type(value: Any, tp: Any, where: str) -> Any:
    """Validate (and lightly coerce int->float) ``value`` against annotation ``tp``."""
    origin = typing.get_origin(tp)
    if origin in (Union, types.UnionType):
        args = typing.get_args(tp)
        if value is None and type(None) in args:
            return None
        non_none = [a for a in args if a is not type(None)]
        return _check_type(value, non_none[0], where)
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{where} must be a list, got {value!r}")
        (inner,) = typing.get_args(tp)
        return [_check_type(v, inner, f"{where}[{i}]") for i, v in enumerate(value)]
    if dataclasses.is_dataclass(tp):
        return _build_section(tp, value, where)  # type: ignore[arg-type]
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where} must be true/false, got {value!r}")
        return value
    if tp is int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ConfigError(f"{where} must be an integer, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where} must be a number, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where} must be a string, got {value!r}")
        return value
    if tp is Path:
        return value
    raise ConfigError(f"Unsupported type {tp} at {where}")  # pragma: no cover


def _build_section(cls: type[T], raw: Any, section: str) -> T:
    """Build dataclass ``cls`` from dict ``raw``; reject missing/unknown keys."""
    if not isinstance(raw, dict):
        raise ConfigError(f"Section '{section}' must be a mapping, got {type(raw).__name__}.")
    hints = typing.get_type_hints(cls)
    expected = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    missing = expected - raw.keys()
    unknown = raw.keys() - expected
    if missing:
        raise ConfigError(f"Section '{section}' is missing keys: {sorted(missing)}")
    if unknown:
        raise ConfigError(f"Section '{section}' has unknown keys: {sorted(unknown)}")
    kwargs = {name: _check_type(raw[name], hints[name], f"{section}.{name}") for name in expected}
    return cls(**kwargs)


def _validate(cfg: Config) -> None:
    """Value checks that types alone cannot express."""
    p = cfg.project
    if p.seed < 0:
        raise ConfigError(f"project.seed must be a non-negative integer, got {p.seed!r}")
    if p.torch_num_threads < 1:
        raise ConfigError("project.torch_num_threads must be an integer >= 1")
    if cfg.logging.level not in VALID_LOG_LEVELS:
        raise ConfigError(f"logging.level must be one of {VALID_LOG_LEVELS}")
    o = cfg.optimizer
    if not 0 < o.max_weight <= o.gross_leverage / 2:
        raise ConfigError("optimizer.max_weight must be in (0, gross_leverage/2]")
    for name in ("sector_limit", "factor_limit", "turnover_limit", "risk_aversion", "solver_eps"):
        if getattr(o, name) <= 0:
            raise ConfigError(f"optimizer.{name} must be > 0")
    if p.device not in ("auto", "cpu", "cuda", "mps"):
        raise ConfigError("project.device must be auto, cpu, cuda or mps")
    am = cfg.alpha_model
    if am.d_model % am.n_heads != 0:
        raise ConfigError("alpha_model.d_model must be divisible by n_heads")
    if am.ensemble_size < 1:
        raise ConfigError("alpha_model.ensemble_size must be >= 1")
    if cfg.risk_model.architecture not in ("lstm", "transformer"):
        raise ConfigError("risk_model.architecture must be 'lstm' or 'transformer'")
    if cfg.risk_model.d_model % cfg.risk_model.n_heads != 0:
        raise ConfigError("risk_model.d_model must be divisible by n_heads")
    if cfg.risk_model.seq_len_days % cfg.risk_model.patch_len_days != 0:
        raise ConfigError("risk_model.seq_len_days must be a multiple of patch_len_days")
    if not 0 < cfg.risk_model.val_fraction < 1:
        raise ConfigError("risk_model.val_fraction must be in (0, 1)")
    if cfg.backtest.transaction_cost_bps < 0:
        raise ConfigError("backtest.transaction_cost_bps must be >= 0")
    if cfg.baselines.hrp_linkage not in ("single", "average", "complete", "ward"):
        raise ConfigError("baselines.hrp_linkage must be single/average/complete/ward")


SECTIONS: dict[str, type] = {
    "project": ProjectConfig,
    "paths": PathsConfig,
    "logging": LoggingConfig,
    "data": DataConfig,
    "signals": SignalsConfig,
    "exposures": ExposuresConfig,
    "backtest": BacktestConfig,
    "alpha": AlphaConfig,
    "baselines": BaselinesConfig,
    "optimizer": OptimizerConfig,
    "risk_model": RiskModelConfig,
    "alpha_model": AlphaModelConfig,
    "e2e": E2EConfig,
    "evaluation": EvaluationConfig,
    "dashboard": DashboardConfig,
}


def find_config(start: Path | None = None) -> Path:
    """Walk up from ``start`` (default: cwd) until ``config.yaml`` is found."""
    here = (start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        candidate = folder / DEFAULT_CONFIG_NAME
        if candidate.is_file():
            return candidate
    raise ConfigError(f"Could not find {DEFAULT_CONFIG_NAME} in {here} or any parent folder.")


def load_config(path: str | Path | None = None) -> Config:
    """Load, type and validate the project configuration.

    Parameters
    ----------
    path:
        Path to a YAML file. If ``None``, search upward from the current
        working directory for ``config.yaml``.
    """
    cfg_path = Path(path).resolve() if path is not None else find_config()
    if not cfg_path.is_file():
        raise ConfigError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigError(f"{cfg_path} must contain a YAML mapping at the top level.")

    missing = SECTIONS.keys() - raw.keys()
    unknown = raw.keys() - SECTIONS.keys()
    if missing:
        raise ConfigError(f"config is missing sections: {sorted(missing)}")
    if unknown:
        raise ConfigError(f"config has unknown sections: {sorted(unknown)}")

    root = cfg_path.parent
    raw_paths = raw["paths"]
    if not isinstance(raw_paths, dict):
        raise ConfigError("Section 'paths' must be a mapping.")
    raw = {**raw, "paths": {k: (root / str(v)).resolve() for k, v in raw_paths.items()}}

    built = {name: _build_section(cls, raw[name], name) for name, cls in SECTIONS.items()}
    cfg = Config(root=root, **built)
    _validate(cfg)
    return cfg


def ensure_directories(cfg: Config) -> None:
    """Create every configured directory if it does not exist yet."""
    for p in cfg.paths.all():
        p.mkdir(parents=True, exist_ok=True)
