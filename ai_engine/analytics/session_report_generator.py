import pandas as pd


class SessionReportGenerator:

    def generate(self, csv_path):

        df = pd.read_csv(csv_path)

        report = {

            "total_records": len(df),

            "avg_emotion_confidence":
                df["emotion_confidence"].mean(),

            "avg_temporal_confidence":
                df["temporal_confidence"].mean(),

            "max_blink_rate":
                df["blink_rate"].max(),

            "dominant_emotion":
                df["emotion"].mode()[0],

            "dominant_temporal_emotion":
                df["temporal_emotion"].mode()[0],

            "most_common_stress":
                df["stress_level"].mode()[0],

            "most_common_cognitive_load":
                df["cognitive_load"].mode()[0],

            "most_common_deception":
                df["deception_risk"].mode()[0],
        }

        return report