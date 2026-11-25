"""Seed the fixed ST-NLP categories when invoked from repository root."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.database.seed import seed_categories

if __name__ == "__main__":
    print(f"Seeded {seed_categories()} missing incident categories.")
