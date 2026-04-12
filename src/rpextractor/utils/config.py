import os
import yaml
import sys
from pathlib import Path
from dotenv import load_dotenv

# 1. DYNAMIC PATH CALCULATION
# __file__ is .../research-paper-extractor/configs/config.py
# .parent is .../configs/
# .parent.parent is the Project Root
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
LOG_DIR = BASE_DIR / "logs"
# 2. THE "IMPORT FIXER"
# If we are running a script directly, the root isn't in sys.path.
# This block injects the root so 'from src.rpextractor...' works everywhere.
if os.getenv("PYTHONPATH") == "." or not os.getenv("PYTHONPATH"):
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))

def load_yaml(file_name: str):
    """Load a YAML config file from the configs directory."""
    config_path = BASE_DIR / "configs" / file_name
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} not found.")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
