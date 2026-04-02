import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 1. DYNAMIC PATH CALCULATION
# __file__ is .../research-paper-extractor/configs/config.py
# .parent is .../configs/
# .parent.parent is the Project Root
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
# 2. THE "IMPORT FIXER"
# If we are running a script directly, the root isn't in sys.path.
# This block injects the root so 'from src.rpextractor...' works everywhere.
if os.getenv("PYTHONPATH") == "." or not os.getenv("PYTHONPATH"):
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
