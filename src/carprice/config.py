"""Configuration loader and typed configuration dataclasses for the car price prediction system."""

from dataclasses import dataclass, field
from functools import lru_cache
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProjectSettings:
    """Project metadata and reproducibility settings."""

    name: str
    version: str
    random_seed: int


@dataclass(frozen=True)
class PathSettings:
    """Resolved file and directory paths across the workspace."""

    root_dir: Path
    raw_data_path: Path
    interim_data_dir: Path
    processed_data_dir: Path
    models_dir: Path
    figures_dir: Path
    tables_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class SplittingSettings:
    """Data split and validation parameters."""

    test_size: float
    cv_folds: int
    stratify_by: str | None


@dataclass(frozen=True)
class CleaningSettings:
    """Thresholds and bounds for data audit and cleaning."""

    km_cap: int
    min_price: int
    max_price: int
    min_year: int
    max_year: int
    max_drop_row_fraction: float


@dataclass(frozen=True)
class FeaturesSettings:
    """Feature engineering and target definition settings."""

    target_column: str
    brand_tier_mapping: dict[str, str]
    rare_brand_threshold: int = 30
    rare_fuel_threshold: int = 40
    luxury_brands: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModelsSettings:
    """Model training, candidate algorithms, and tuning configurations."""

    candidates: list[str]
    tuning_trials: int
    timeout_seconds: int


@dataclass(frozen=True)
class AppConfig:
    """Root configuration holding all sub-configurations."""

    project: ProjectSettings
    paths: PathSettings
    splitting: SplittingSettings
    cleaning: CleaningSettings
    features: FeaturesSettings
    models: ModelsSettings


def find_project_root() -> Path:
    """Locate the project root directory based on repository structure.

    Returns:
        Path: Absolute path to the repository root directory.
    """
    return Path(__file__).resolve().parent.parent.parent


def _build_paths(root: Path, raw_paths: dict[str, Any]) -> PathSettings:
    """Construct resolved PathSettings from raw path strings.

    Args:
        root: Workspace root directory path.
        raw_paths: Dictionary of relative path definitions.

    Returns:
        PathSettings: Strongly-typed resolved paths.
    """
    return PathSettings(
        root_dir=root,
        raw_data_path=root / raw_paths["raw_data_path"],
        interim_data_dir=root / raw_paths["interim_data_dir"],
        processed_data_dir=root / raw_paths["processed_data_dir"],
        models_dir=root / raw_paths["models_dir"],
        figures_dir=root / raw_paths["figures_dir"],
        tables_dir=root / raw_paths["tables_dir"],
        reports_dir=root / raw_paths["reports_dir"],
    )


@lru_cache(maxsize=1)
def load_config(config_path: Path | str | None = None) -> AppConfig:
    """Load configuration YAML and return a typed AppConfig instance.

    Args:
        config_path: Optional explicit path to config YAML file.

    Returns:
        AppConfig: Frozen configuration object.

    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
        yaml.YAMLError: If parsing configuration YAML fails.
    """
    root = find_project_root()
    path = Path(config_path) if config_path else root / "configs" / "config.yaml"

    if not path.is_absolute():
        path = root / path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    logger.debug("Loaded configuration from %s", path)

    return AppConfig(
        project=ProjectSettings(**data["project"]),
        paths=_build_paths(root, data["paths"]),
        splitting=SplittingSettings(**data["splitting"]),
        cleaning=CleaningSettings(**data["cleaning"]),
        features=FeaturesSettings(**data["features"]),
        models=ModelsSettings(**data["models"]),
    )


def get_config() -> AppConfig:
    """Retrieve the cached singleton application configuration.

    Returns:
        AppConfig: Application configuration instance.
    """
    return load_config()
