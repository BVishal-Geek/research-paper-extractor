"""Path resolution and YAML config loading for the rpextractor package."""

import os
import sys
from pathlib import Path

import yaml

# BASE_DIR resolves to the project root by walking up from this file.
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
LOG_DIR = BASE_DIR / "logs"

# When a script is invoked directly, the project root isn't on sys.path.
# Inject it so `from rpextractor...` imports work regardless of entry point.
if os.getenv("PYTHONPATH") == "." or not os.getenv("PYTHONPATH"):
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))


def load_yaml(file_name: str):
    """Load a YAML config file from the configs directory."""
    config_path = BASE_DIR / "configs" / file_name
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} not found.")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
