"""Tests for strict config loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dpc.config import ConfigError, ensure_directories, find_config, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


@pytest.fixture
def raw_config() -> dict:
    """The real project config as a plain dict, so tests track it automatically."""
    return yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))


def test_real_config_loads() -> None:
    cfg = load_config(PROJECT_ROOT / "config.yaml")
    assert cfg.project.seed >= 0
    assert cfg.root == PROJECT_ROOT
    # every path is absolute and lives inside the project
    for p in cfg.paths.all():
        assert p.is_absolute()
        assert PROJECT_ROOT in p.parents


def test_config_is_immutable() -> None:
    cfg = load_config(PROJECT_ROOT / "config.yaml")
    with pytest.raises(AttributeError):
        cfg.project.seed = 7  # type: ignore[misc]


def test_unknown_key_is_rejected(tmp_path: Path, raw_config: dict) -> None:
    raw_config["project"]["seeed"] = 1  # typo must not be silently ignored
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(_write(tmp_path, raw_config))


def test_missing_key_is_rejected(tmp_path: Path, raw_config: dict) -> None:
    del raw_config["project"]["seed"]  # no silent default seed
    with pytest.raises(ConfigError, match="missing keys"):
        load_config(_write(tmp_path, raw_config))


def test_missing_section_is_rejected(tmp_path: Path, raw_config: dict) -> None:
    del raw_config["paths"]
    with pytest.raises(ConfigError, match="missing sections"):
        load_config(_write(tmp_path, raw_config))


@pytest.mark.parametrize("bad_seed", [-1, 1.5, "42", True])
def test_invalid_seed_is_rejected(tmp_path: Path, raw_config: dict, bad_seed: object) -> None:
    raw_config["project"]["seed"] = bad_seed
    with pytest.raises(ConfigError, match="seed"):
        load_config(_write(tmp_path, raw_config))


def test_invalid_log_level_is_rejected(tmp_path: Path, raw_config: dict) -> None:
    raw_config["logging"]["level"] = "LOUD"
    with pytest.raises(ConfigError, match="logging.level"):
        load_config(_write(tmp_path, raw_config))


def test_paths_resolve_relative_to_config_file(tmp_path: Path, raw_config: dict) -> None:
    cfg = load_config(_write(tmp_path, raw_config))
    assert cfg.paths.data_raw == (tmp_path / "data" / "raw").resolve()
    ensure_directories(cfg)
    assert all(p.is_dir() for p in cfg.paths.all())


def test_find_config_walks_up(tmp_path: Path, raw_config: dict) -> None:
    cfg_path = _write(tmp_path, raw_config)
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert find_config(nested) == cfg_path.resolve()
