"""
Phase 7 - Session Recording.

Every processed frame is stored so a full session can be replayed and, later
(Phase 8), turned into an automatic report.

For each frame the dissertation's core signals are captured:

    timestamp, emotion, stress, audio_emotion, audio_stress,
    blink, movement, gaze, risk, confidence

Two persisted artefacts are produced per session, sharing the same stem
(``session_<YYYYMMDD_HHMMSS>``):

    * a flat CSV   - spreadsheet / quick inspection (kept for backward
      compatibility with the earlier phases).
    * a structured JSON - session metadata plus an ordered list of frame
      records. This is the authoritative input for the report generator.

The recorder is stateful: a live session calls :meth:`reset` on connect so
frames from a previous session never leak into the current recording.
"""

import csv
import json
import os
from datetime import datetime


class SessionLogger:
    """Records every frame of a session and exports it as CSV and JSON."""

    def __init__(self, output_dir="outputs/session_logs"):

        self.output_dir = output_dir

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        self.reset()

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        """Begin a fresh recording. Called when a new live session starts."""

        # Full per-frame results (source for the flat CSV export).
        self.records = []

        # Structured Phase-7 frame records (source for the JSON export).
        self.frames = []

        self.session_id = (
            f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        self.started_at = None

        # datetime of the first recorded frame, used to compute elapsed time.
        self._start_ts = None

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def add_record(self, result):
        """Store one processed frame.

        ``result`` is the merged visual + audio + fusion dictionary produced
        per frame by the websocket pipeline. The dict is stamped with a
        timestamp (used by the CSV) and projected onto the structured frame
        schema (used by the JSON).
        """

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
        """Project a raw fusion result onto the Phase-7 frame schema.

        Only the dissertation's core signals are kept, and grouped so the
        report generator can consume them without knowing the internal key
        names of the fusion engine.
        """

        elapsed = (
            round((now - self._start_ts).total_seconds(), 2)
            if self._start_ts is not None
            else 0.0
        )

        return {

            "timestamp": result["timestamp"],

            # Seconds since the first frame of the session.
            "elapsed": elapsed,

            # VISUAL (face) emotion is the headline the report/LLM consume - it
            # is the trustworthy channel. The multimodal-fused emotion is kept
            # alongside as a secondary field for transparency.
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

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self):
        """Lightweight snapshot of the current recording (for the UI)."""

        return {
            "session_id": self.session_id,
            "recording": self._start_ts is not None,
            "started_at": self.started_at,
            "frame_count": len(self.frames),
        }

    def snapshot(self):
        """Return the current recording as a session dict (no disk I/O).

        Same shape as the persisted JSON export, so the Phase-8 report
        generator can consume the live in-progress recording directly.
        """

        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "frame_count": len(self.frames),
            "frames": self.frames,
        }

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------

    def export_json(self):
        """Write the structured session recording to disk.

        Returns the file path, or ``None`` when nothing has been recorded.
        """

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
        """Write the flat per-frame log to disk (backward compatible)."""

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
