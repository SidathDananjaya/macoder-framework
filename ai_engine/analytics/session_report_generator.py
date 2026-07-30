import json
from collections import Counter


STRESS_PERCENT = {
    "LOW": 0.0,
    "MEDIUM": 50.0,
    "HIGH": 100.0,
}

STRESS_HIGH_THRESHOLD = 66.0
STRESS_MEDIUM_THRESHOLD = 33.0

RISK_HIGH_AVG = 70.0
RISK_HIGH_PEAK = 85.0
RISK_MODERATE_AVG = 40.0
GAZE_AVOIDANCE_HIGH = 40.0
EMOTION_STABILITY_LOW = 50.0
BLINK_RATE_HIGH = 30.0


def _fmt_time(seconds):
    seconds = int(round(seconds or 0))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _count_changes(labels):
    return sum(1 for prev, cur in zip(labels, labels[1:]) if prev != cur)


def _mode(labels, default="unknown"):
    if not labels:
        return default
    return Counter(labels).most_common(1)[0][0]


class SessionReportGenerator:
    def generate(self, session):
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


    def _load(self, session):
        if isinstance(session, str):
            try:
                with open(session, "r", encoding="utf-8") as file:
                    return json.load(file)
            except (OSError, ValueError):
                return {}

        return session or {}

    def _duration(self, frames):
        return frames[-1].get("elapsed", 0.0) or 0.0


    def _stress_section(self, frames):
        levels = [f.get("stress", "LOW") for f in frames]
        percents = [STRESS_PERCENT.get(level, 0.0) for level in levels]

        average_percent = round(sum(percents) / len(percents), 1)

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
        emotions = [f.get("emotion", "neutral") for f in frames]

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
        counts = [f.get("blink", 0) or 0 for f in frames]
        total = max(counts)

        minutes = duration / 60.0
        per_minute = round(total / minutes, 1) if minutes > 0 else 0.0

        return {
            "total": total,
            "per_minute": per_minute,
        }

    def _gaze_section(self, frames):
        stabilities = [
            f.get("gaze", {}).get("stability", 1.0) for f in frames
        ]
        avg_stability = sum(stabilities) / len(stabilities)

        stability_percent = round(avg_stability * 100, 1)

        return {
            "average_stability_percent": stability_percent,
            "avoidance_percent": round(100.0 - stability_percent, 1),
        }


    def _summary(self, report):
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


    def _risk_score(self, frame):
        return frame.get("risk", {}).get("score", 0) or 0

    def _stress_label(self, percent):
        if percent >= STRESS_HIGH_THRESHOLD:
            return "HIGH"
        if percent >= STRESS_MEDIUM_THRESHOLD:
            return "MEDIUM"
        return "LOW"

    def _distribution(self, labels, order=None):
        counts = Counter(labels)
        if order is not None:
            return {key: counts.get(key, 0) for key in order}
        return dict(counts.most_common())

    def _change_timeline(self, frames, value, build, carry=None):
        timeline = []
        previous = object()

        for frame in frames:
            current = value(frame)
            if current != previous:
                time = _fmt_time(frame.get("elapsed", 0.0))
                payload = (current, carry(frame)) if carry else current
                timeline.append(build(time, payload))
                previous = current

        return timeline

    def _empty_report(self, session):
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
