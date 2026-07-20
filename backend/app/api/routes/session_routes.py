from fastapi import APIRouter
import pandas as pd
import os
import glob

router = APIRouter()

# Project root directory
BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../../.."
    )
)

# outputs/session_logs path
SESSION_LOG_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "session_logs"
)

print("SESSION LOG PATH:")
print(SESSION_LOG_DIR)

def latest_session_file():

    files = glob.glob(
        os.path.join(SESSION_LOG_DIR, "*.csv")
    )

    print("FOUND FILES:")
    print(files)

    if not files:
        return None

    latest = max(
        files,
        key=os.path.getctime
    )

    return latest


@router.get("/session-data")
def get_session_data():

    file = latest_session_file()

    if file is None:
        return []

    df = pd.read_csv(file)

    # Optional cleanup for NaN values
    df = df.fillna("")

    # Convert dataframe to JSON
    return df.to_dict(
        orient="records"
    )


# The session CSV schema has evolved across phases, so read columns
# defensively: pick the first column that exists and degrade to a default
# rather than raising a KeyError (which surfaced as ASGI 500s on the dashboard's
# summary tiles).

def _mean_of(df, *names, default=0.0):
    for name in names:
        if name in df.columns:
            value = pd.to_numeric(df[name], errors="coerce").mean()
            return round(float(value), 2) if pd.notna(value) else default
    return default


def _max_of(df, *names, default=0.0):
    for name in names:
        if name in df.columns:
            value = pd.to_numeric(df[name], errors="coerce").max()
            return round(float(value), 2) if pd.notna(value) else default
    return default


def _mode_of(df, *names, default="unknown"):
    for name in names:
        if name in df.columns and not df[name].dropna().empty:
            return df[name].mode()[0]
    return default


@router.get("/session-summary")
def get_session_summary():

    file = latest_session_file()

    if file is None:
        return {
            "error": "No session CSV files found"
        }

    df = pd.read_csv(file)

    if len(df) == 0:
        return {"error": "Session CSV is empty"}

    return {
        "records": len(df),

        # Column names differ across phases; fall back through the likely names.
        "avg_emotion_confidence":
            _mean_of(df, "cognitive_confidence", "emotion_confidence"),

        "avg_temporal_confidence":
            _mean_of(df, "fusion_confidence", "temporal_confidence"),

        "max_blink_rate":
            _max_of(df, "blink_count", "blink_rate"),

        "dominant_emotion":
            _mode_of(df, "final_emotion", "emotion"),

        "dominant_temporal_emotion":
            _mode_of(df, "temporal_emotion", "emotion"),

        "most_common_stress":
            _mode_of(df, "stress_level"),

        "most_common_cognitive_load":
            _mode_of(df, "cognitive_load"),

        "most_common_deception":
            _mode_of(df, "deception_risk"),
    }