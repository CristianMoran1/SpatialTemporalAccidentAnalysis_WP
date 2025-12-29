import sys
from pathlib import Path

# Add the repo root to sys.path so "import app_utils" works
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
