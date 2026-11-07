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


def test_split_and_encoder_settings_come_from_config() -> None:
    """Split stratification and rare-category threshold are explicit config values."""
    config = get_config()
    assert config.splitting.stratify_by == "selling_price"
    assert config.splitting.stratify_bins == 5
    assert config.features.rare_category_min_frequency == 20
    assert config.eda.rare_brand_report_threshold == 30


def test_every_tuned_model_has_a_search_space() -> None:
    """Each search space belongs to a configured candidate and defines bounded parameters."""
    models = get_config().models
    assert set(models.search_spaces) <= set(models.candidates)
    for space in models.search_spaces.values():
        for spec in space.values():
            assert spec["type"] in {"int", "float"}
            assert spec["low"] < spec["high"]
    assert set(models.candidate_params) == set(models.candidates)


def test_evaluation_settings_are_sane() -> None:
    """Phase 5 evaluation settings load and are internally consistent (D-023)."""
    evaluation = get_config().evaluation
    assert 0 < evaluation.interval_nominal_coverage < 1
    assert 0 < evaluation.confidence_level < 1
    assert evaluation.within_tolerances == sorted(evaluation.within_tolerances)
    assert evaluation.age_band_edges == sorted(evaluation.age_band_edges)
    assert evaluation.bootstrap_resamples >= 100
