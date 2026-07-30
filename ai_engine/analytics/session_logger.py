import csv
import json
import os
from datetime import datetime


class SessionLogger:
    def __init__(self, output_dir="outputs/session_logs"):

        self.output_dir = output_dir

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        self.reset()


    def reset(self):
        self.records = []

        self.frames = []

        self.session_id = (
            f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        self.started_at = None

        self._start_ts = None


    def add_record(self, result):
        now = datetime.now()

        if self._start_ts is None:
            self._start_ts = now
            self.started_at = now.strftime("%Y-%m-%d %H:%M:%S")

        result["timestamp"] = now.strftime("%Y-%m-%d %H:%M:%S")

        self.records.append(result)

        self.frames.append(
            self._build_frame(result, now)
        )

    def _build_frame(self, result, now):
        elapsed = (
            round((now - self._start_ts).total_seconds(), 2)
            if self._start_ts is not None
            else 0.0
        )

        return {

            "timestamp": result["timestamp"],

            "elapsed": elapsed,

            "emotion": result.get(
                "emotion",
                result.get("final_emotion", "neutral")
            ),
            "visual_emotion": result.get("emotion", "neutral"),
            "fused_emotion": result.get(
                "fused_emotion",
                result.get("emotion", "neutral")
            ),

            "stress": result.get(
                "stress_level",
                result.get("fused_stress", "LOW")
            ),

            "audio_emotion": result.get("audio_emotion", "neutral"),
            "audio_stress": result.get("audio_stress", "LOW"),

            "blink": result.get("blink_count", 0),
            "movement": result.get("movement_score", 0),

            "gaze": {
                "direction": result.get("gaze_direction", "CENTER"),
                "stability": result.get("gaze_stability", 1.0),
                "shift_frequency": result.get("gaze_shift_frequency", 0),
            },

            "risk": {
                "score": result.get("risk_score", 0),
                "level": result.get("risk_level", "LOW"),
                "deception": result.get("deception_risk", "LOW"),
            },

            "cognitive_load": result.get("cognitive_load", "LOW"),
            "fusion_score": result.get("fusion_score", 0),

            "confidence": result.get("fusion_confidence", 0.0),
        }


    def status(self):
        return {
            "session_id": self.session_id,
            "recording": self._start_ts is not None,
            "started_at": self.started_at,
            "frame_count": len(self.frames),
        }

    def snapshot(self):
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "frame_count": len(self.frames),
            "frames": self.frames,
        }


    def export_json(self):
        if len(self.frames) == 0:
            return None

        session = {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "frame_count": len(self.frames),
            "frames": self.frames,
        }

        filename = os.path.join(
            self.output_dir,
            f"{self.session_id}.json"
        )

        with open(
            filename,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(session, file, indent=2)

        return filename

    def export_csv(self):
        if len(self.records) == 0:
            return None

        filename = os.path.join(
            self.output_dir,
            f"{self.session_id}.csv"
        )

        keys = self.records[0].keys()

        with open(
            filename,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=keys,
                extrasaction="ignore"
            )

            writer.writeheader()

            writer.writerows(self.records)

        return filename
