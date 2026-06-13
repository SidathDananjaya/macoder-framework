import csv
from pathlib import Path
from datetime import datetime


class SessionLogger:

    def __init__(self):

        self.session_id = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        self.output_dir = Path(
            "outputs/session_logs"
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.file_path = (
            self.output_dir /
            f"session_{self.session_id}.csv"
        )

        self.initialize_csv()

    def initialize_csv(self):

        with open(
            self.file_path,
            mode="w",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([

                "timestamp",

                "emotion",
                "emotion_confidence",

                "temporal_emotion",
                "temporal_confidence",

                "stress_level",

                "cognitive_load",

                "deception_risk",

                "gaze",

                "yaw",
                "pitch",
                "roll",

                "blink_rate",

                "anomaly_risk",

                "fusion_confidence",

                "final_state",

                "cognitive_assessment"
            ])

    def log(self, data):

        with open(
            self.file_path,
            mode="a",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([

                datetime.now().strftime(
                    "%H:%M:%S"
                ),

                data["emotion"],
                data["emotion_confidence"],

                data["temporal_emotion"],
                data["temporal_confidence"],

                data["stress_level"],

                data["cognitive_load"],

                data["deception_risk"],

                data["gaze"],

                data["yaw"],
                data["pitch"],
                data["roll"],

                data["blink_rate"],

                data["anomaly_risk"],

                data["fusion_confidence"],

                data["final_state"],

                data["cognitive_assessment"]
            ])