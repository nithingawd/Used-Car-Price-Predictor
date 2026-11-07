"""Unit tests for configuration loading and validation."""

from pathlib import Path

from carprice.config import AppConfig, get_config, load_config


def test_load_config_structure() -> None:
    """Verify that configuration loads successfully into typed dataclass."""
    config = load_config()
    assert isinstance(config, AppConfig)
    assert config.project.name == "used-car-price-predictor"
    assert config.project.random_seed == 42
    assert config.splitting.test_size == 0.2
    assert config.splitting.cv_folds == 5
    assert "xgboost" in config.models.candidates


def test_all_configured_paths_resolve() -> None:
    """Ensure that all directory paths and raw data file exist on disk."""
    config = get_config()
    paths = config.paths

    assert isinstance(paths.root_dir, Path)
    assert paths.root_dir.exists()
    assert paths.raw_data_path.is_file()
    assert paths.raw_data_path.exists()
    assert paths.interim_data_dir.is_dir()
    assert paths.processed_data_dir.is_dir()
    assert paths.models_dir.is_dir()
    assert paths.figures_dir.is_dir()
    assert paths.tables_dir.is_dir()
    assert paths.reports_dir.is_dir()


def test_get_config_is_cached() -> None:
    """Verify that repeated get_config calls return the exact same instance."""
    cfg1 = get_config()
    cfg2 = get_config()
    assert cfg1 is cfg2
