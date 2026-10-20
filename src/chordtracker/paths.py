"""Repo-anchored paths, so code works the same from a notebook, a script or a test."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]    # src/chordtracker/paths.py -> repo root
DATA_DIR = REPO_ROOT / "data"                       # git-ignored: datasets and everything generated from them
SPLITS_DIR = REPO_ROOT / "splits"                   # tracked: one split file per dataset
