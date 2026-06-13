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


@router.get("/session-summary")
def get_session_summary():

    file = latest_session_file()

    if file is None:
        return {
            "error": "No session CSV files found"
        }

    df = pd.read_csv(file)

    return {
        "records": len(df),

        "avg_emotion_confidence":
            round(df["emotion_confidence"].mean(), 2),

        "avg_temporal_confidence":
            round(df["temporal_confidence"].mean(), 2),

        "max_blink_rate":
            round(df["blink_rate"].max(), 2),

        "dominant_emotion":
            df["emotion"].mode()[0],

        "dominant_temporal_emotion":
            df["temporal_emotion"].mode()[0],

        "most_common_stress":
            df["stress_level"].mode()[0],

        "most_common_cognitive_load":
            df["cognitive_load"].mode()[0],

        "most_common_deception":
            df["deception_risk"].mode()[0]
    }