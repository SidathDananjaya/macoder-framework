"""
Phase 8 - Automatic Report Generator.

At the end of a session the per-frame recording produced in Phase 7 is turned
into a single, human-readable **interview report**. The report is the analytical
pay-off of the whole pipeline and is designed to be the strongest, most
defensible artefact of the dissertation.

Authoritative input
-------------------
The Phase-7 structured JSON recording (``SessionLogger.export_json``), i.e. a
session dict of the form::

    {
        "session_id": "...",
        "started_at": "...",
        "ended_at": "...",
        "frame_count": N,
        "frames": [ <frame>, ... ]
    }

where each ``<frame>`` follows the schema built by
``SessionLogger._build_frame`` (elapsed, emotion, stress, audio_*, blink,
movement, gaze{...}, risk{...}, cognitive_load, confidence).

Output
------
A structured report dict with the sections requested by the specification:

    summary, stress (average / highest), emotion timeline, risk timeline,
    blink statistics, gaze statistics and rule-based recommendations.

Everything is rule-based and transparent - the free-text *interpretation* of the
session is deliberately left to Phase 9 (LLM Interpretation), which consumes this
report as its input.
"""

import json
from collections import Counter


# Categorical stress -> percentage intensity. Mirrors the Risk Engine's
# ``STRESS_MAP`` (LOW 0.0 / MEDIUM 0.5 / HIGH 1.0), expressed on a 0-100 scale so
# an "average stress" can be reported as a percentage.
STRESS_PERCENT = {
    "LOW": 0.0,
    "MEDIUM": 50.0,
    "HIGH": 100.0,
}

# Percentage thresholds for turning an average stress percentage back into a
# categorical label.
STRESS_HIGH_THRESHOLD = 66.0
STRESS_MEDIUM_THRESHOLD = 33.0

# Recommendation thresholds (documented so they can be defended in the write-up).
RISK_HIGH_AVG = 70.0          # avg risk score at/above this -> high concern
RISK_HIGH_PEAK = 85.0         # a single peak at/above this -> high concern
RISK_MODERATE_AVG = 40.0      # avg risk score at/above this -> moderate concern
GAZE_AVOIDANCE_HIGH = 40.0    # % of gaze aversion considered notable
EMOTION_STABILITY_LOW = 50.0  # emotion stability (%) below this is "volatile"
BLINK_RATE_HIGH = 30.0        # blinks/minute above this is elevated


def _fmt_time(seconds):
    """Elapsed seconds -> ``MM:SS`` (matches the live Timeline formatting)."""
    seconds = int(round(seconds or 0))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _count_changes(labels):
    """Number of times a label differs from the previous one."""
    return sum(1 for prev, cur in zip(labels, labels[1:]) if prev != cur)


def _mode(labels, default="unknown"):
    """Most common label, or ``default`` for an empty sequence."""
    if not labels:
        return default
    return Counter(labels).most_common(1)[0][0]


class SessionReportGenerator:
    """Turns a Phase-7 session recording into a structured interview report."""

    def generate(self, session):
        """Build the report.

        ``session`` may be:

        * a session dict (as produced by ``SessionLogger.export_json`` /
          held in memory as ``{"frames": [...], ...}``), or
        * a path (str) to a stored JSON recording.

        An empty / missing recording yields a well-formed "no data" report
        rather than raising, so callers can render it unconditionally.
        """

        session = self._load(session)
        frames = session.get("frames", []) if session else []

        if not frames:
            return self._empty_report(session)

        duration = self._duration(frames)

        stress = self._stress_section(frames)
        emotion = self._emotion_section(frames)
        risk = self._risk_section(frames)
        blink = self._blink_section(frames, duration)
        gaze = self._gaze_section(frames)

        report = {
            "session_id": session.get("session_id", "unknown"),
            "started_at": session.get("started_at"),
            "ended_at": session.get("ended_at"),
            "frame_count": len(frames),
            "duration_seconds": round(duration, 1),
            "duration": _fmt_time(duration),

            "stress": stress,
            "emotion": emotion,
            "risk": risk,
            "blink": blink,
            "gaze": gaze,
        }

        report["summary"] = self._summary(report)
        report["recommendations"] = self._recommendations(report)

        return report

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def _load(self, session):
        """Accept a dict directly, or load a JSON recording from a path."""

        if isinstance(session, str):
            try:
                with open(session, "r", encoding="utf-8") as file:
                    return json.load(file)
            except (OSError, ValueError):
                return {}

        return session or {}

    def _duration(self, frames):
        """Session length in seconds, from the last frame's elapsed time."""
        return frames[-1].get("elapsed", 0.0) or 0.0

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def _stress_section(self, frames):
        """Average and highest stress, plus the level distribution."""

        levels = [f.get("stress", "LOW") for f in frames]
        percents = [STRESS_PERCENT.get(level, 0.0) for level in levels]

        average_percent = round(sum(percents) / len(percents), 1)

        # Highest = first frame that reached the maximum observed intensity.
        peak_percent = max(percents)
        peak_index = percents.index(peak_percent)

        return {
            "average_percent": average_percent,
            "average_level": self._stress_label(average_percent),
            "highest_level": levels[peak_index],
            "highest_at": _fmt_time(frames[peak_index].get("elapsed", 0.0)),
            "distribution": self._distribution(
                levels, order=["LOW", "MEDIUM", "HIGH"]
            ),
        }

    def _emotion_section(self, frames):
        """Dominant emotion, distribution, stability and change timeline."""

        emotions = [f.get("emotion", "neutral") for f in frames]

        # Stability: 100% = never changed, 0% = changed every frame.
        if len(emotions) > 1:
            change_rate = _count_changes(emotions) / (len(emotions) - 1)
            stability = round((1.0 - change_rate) * 100, 1)
        else:
            stability = 100.0

        return {
            "dominant": _mode(emotions, "neutral"),
            "distribution": self._distribution(emotions),
            "stability_percent": stability,
            "timeline": self._change_timeline(
                frames,
                value=lambda f: f.get("emotion", "neutral"),
                build=lambda time, emo: {"time": time, "emotion": emo},
            ),
        }

    def _risk_section(self, frames):
        """Average / highest risk score, dominant level and level timeline."""

        scores = [self._risk_score(f) for f in frames]
        levels = [f.get("risk", {}).get("level", "LOW") for f in frames]

        peak_score = max(scores)
        peak_index = scores.index(peak_score)

        return {
            "average_score": round(sum(scores) / len(scores), 1),
            "highest_score": round(peak_score, 1),
            "highest_at": _fmt_time(frames[peak_index].get("elapsed", 0.0)),
            "dominant_level": _mode(levels, "LOW"),
            "final_deception": frames[-1].get("risk", {}).get(
                "deception", "LOW"
            ),
            "distribution": self._distribution(
                levels, order=["LOW", "MEDIUM", "HIGH"]
            ),
            "timeline": self._change_timeline(
                frames,
                value=lambda f: f.get("risk", {}).get("level", "LOW"),
                build=lambda time, f_level: {
                    "time": time,
                    "level": f_level[0],
                    "score": round(f_level[1], 1),
                },
                carry=lambda f: self._risk_score(f),
            ),
        }

    def _blink_section(self, frames, duration):
        """Blink totals and rate.

        ``blink`` is the cumulative blink count captured per frame, so the total
        for the session is its maximum value.
        """

        counts = [f.get("blink", 0) or 0 for f in frames]
        total = max(counts)

        minutes = duration / 60.0
        per_minute = round(total / minutes, 1) if minutes > 0 else 0.0

        return {
            "total": total,
            "per_minute": per_minute,
        }

    def _gaze_section(self, frames):
        """Average gaze stability and its complement (eye avoidance)."""

        stabilities = [
            f.get("gaze", {}).get("stability", 1.0) for f in frames
        ]
        avg_stability = sum(stabilities) / len(stabilities)

        stability_percent = round(avg_stability * 100, 1)

        return {
            "average_stability_percent": stability_percent,
            "avoidance_percent": round(100.0 - stability_percent, 1),
        }

    # ------------------------------------------------------------------
    # Narrative + recommendations
    # ------------------------------------------------------------------

    def _summary(self, report):
        """One-paragraph, headline summary of the session."""

        emotion = report["emotion"]
        stress = report["stress"]
        risk = report["risk"]

        dominant_share = report["emotion"]["distribution"].get(
            emotion["dominant"], 0
        )
        share_percent = round(
            dominant_share / report["frame_count"] * 100
        )

        return (
            f"Candidate appeared '{emotion['dominant']}' for about "
            f"{share_percent}% of the {report['duration']} session. "
            f"Average stress was {stress['average_level']} "
            f"({stress['average_percent']}%), peaking at "
            f"{stress['highest_level']} around {stress['highest_at']}. "
            f"Overall deception risk was {risk['dominant_level']} "
            f"(avg {risk['average_score']}/100, peak "
            f"{risk['highest_score']} at {risk['highest_at']})."
        )

    def _recommendations(self, report):
        """Rule-based recommendations, ordered most-serious first."""

        recs = []

        risk = report["risk"]
        stress = report["stress"]
        gaze = report["gaze"]
        emotion = report["emotion"]
        blink = report["blink"]

        if (
            risk["average_score"] >= RISK_HIGH_AVG
            or risk["highest_score"] >= RISK_HIGH_PEAK
        ):
            recs.append(
                "Elevated deception risk detected "
                f"(avg {risk['average_score']}/100, peak "
                f"{risk['highest_score']}). Recommend a follow-up interview "
                "and manual review of the high-risk moments in the timeline."
            )
        elif risk["average_score"] >= RISK_MODERATE_AVG:
            recs.append(
                "Moderate risk indicators present. Review the flagged "
                "segments before drawing conclusions."
            )

        if stress["average_level"] == "HIGH":
            recs.append(
                "Candidate showed sustained high stress. Consider whether "
                "question difficulty or phrasing contributed."
            )

        if gaze["avoidance_percent"] >= GAZE_AVOIDANCE_HIGH:
            recs.append(
                "Frequent gaze aversion observed "
                f"({gaze['avoidance_percent']}% of the session), which may "
                "indicate discomfort or evasion."
            )

        if emotion["stability_percent"] < EMOTION_STABILITY_LOW:
            recs.append(
                "Emotional state was volatile "
                f"({emotion['stability_percent']}% stability); emotions "
                "shifted frequently across the session."
            )

        if blink["per_minute"] >= BLINK_RATE_HIGH:
            recs.append(
                f"Elevated blink rate ({blink['per_minute']}/min), a "
                "possible stress indicator."
            )

        if not recs:
            recs.append(
                "Candidate remained calm and stable throughout, with low "
                "deception indicators across all modalities."
            )

        return recs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _risk_score(self, frame):
        """Numeric risk score for a frame (0..100), robust to missing keys."""
        return frame.get("risk", {}).get("score", 0) or 0

    def _stress_label(self, percent):
        """Average stress percentage -> categorical label."""
        if percent >= STRESS_HIGH_THRESHOLD:
            return "HIGH"
        if percent >= STRESS_MEDIUM_THRESHOLD:
            return "MEDIUM"
        return "LOW"

    def _distribution(self, labels, order=None):
        """Count occurrences of each label.

        When ``order`` is given the result is an ordered dict over exactly those
        keys (missing ones reported as 0); otherwise it is ordered by frequency.
        """
        counts = Counter(labels)
        if order is not None:
            return {key: counts.get(key, 0) for key in order}
        return dict(counts.most_common())

    def _change_timeline(self, frames, value, build, carry=None):
        """Collapse a per-frame series into change-points.

        A new entry is emitted only when ``value(frame)`` differs from the
        previous frame's value (matching how the live Timeline records changes).

        ``build`` receives ``(time, payload)`` where ``payload`` is the value
        itself, or - when ``carry`` is supplied - the tuple
        ``(value, carry(frame))`` so extra context (e.g. the risk score at the
        moment of a level change) can be attached.
        """

        timeline = []
        previous = object()  # sentinel: guarantees the first frame emits.

        for frame in frames:
            current = value(frame)
            if current != previous:
                time = _fmt_time(frame.get("elapsed", 0.0))
                payload = (current, carry(frame)) if carry else current
                timeline.append(build(time, payload))
                previous = current

        return timeline

    def _empty_report(self, session):
        """Well-formed report for a session with no recorded frames."""
        return {
            "session_id": (session or {}).get("session_id", "unknown"),
            "started_at": (session or {}).get("started_at"),
            "ended_at": (session or {}).get("ended_at"),
            "frame_count": 0,
            "duration_seconds": 0.0,
            "duration": "00:00",
            "stress": {},
            "emotion": {},
            "risk": {},
            "blink": {},
            "gaze": {},
            "summary": "No frames were recorded for this session.",
            "recommendations": [
                "No data captured - start a live session before generating "
                "a report."
            ],
        }
