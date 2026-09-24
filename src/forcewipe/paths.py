"""Locations of repository assets and separately supplied training collections."""
from pathlib import Path
import os

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DATA_ROOT = Path(os.environ.get(
    "FORCEWIPE_TRAINING_DATA_ROOT", REPOSITORY_ROOT / "data/training_collections"
))
