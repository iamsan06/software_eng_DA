"""Configuration loader and management for Phase 1."""
from pathlib import Path
import yaml
from typing import Any, Dict

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "phase1.yaml"


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load and validate the phase 1 YAML configuration file."""
    if config_path is None:
        path = DEFAULT_CONFIG_PATH
    else:
        path = Path(config_path)

    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolve project root relative paths
    root_dir = path.resolve().parent.parent
    config["paths"]["root_dir"] = str(root_dir)

    return config
