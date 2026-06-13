from pathlib import Path

# Root project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset paths
DATASETS_DIR = PROJECT_ROOT / "datasets"

RAW_DATASETS_DIR = DATASETS_DIR / "raw"

RAVDESS_PATH = (
    RAW_DATASETS_DIR /
    "RAVDESS" /
    "Audio_Speech_Actors_01-24"
)

# Processed data
PROCESSED_DIR = DATASETS_DIR / "processed"

# Metadata
METADATA_DIR = DATASETS_DIR / "metadata"

# Experiment outputs
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"