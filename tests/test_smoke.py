"""Smoke tests for the project foundation."""

from pathlib import Path

import aic2026


def test_package_imports():
    """The package can be imported and exposes a version string."""
    assert isinstance(aic2026.__version__, str)
    assert aic2026.__version__


def test_default_config_exists():
    """The default configuration file exists in the repository."""
    config_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    assert config_path.is_file()
    assert config_path.stat().st_size > 0