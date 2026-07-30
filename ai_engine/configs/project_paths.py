from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS_DIR = PROJECT_ROOT / "datasets"

RAW_DATASETS_DIR = DATASETS_DIR / "raw"

RAVDESS_PATH = (
    RAW_DATASETS_DIR /
    "RAVDESS" /
    "Audio_Speech_Actors_01-24"
)

PROCESSED_DIR = DATASETS_DIR / "processed"

METADATA_DIR = DATASETS_DIR / "metadata"

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
