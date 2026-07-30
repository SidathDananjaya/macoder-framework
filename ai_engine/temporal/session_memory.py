import time
from collections import Counter, deque

from ai_engine.analytics.session_trend_analyzer import (
    SessionTrendAnalyzer
)


STRESS_TREND_THRESHOLD = 0.05

FUSION_DELTA_THRESHOLD = 3.0


def _mode(values):
    if not values:
        return "unknown"
    return Counter(values).most_common(1)[0][0]


def _mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def _count_changes(labels):
    changes = 0
    for prev, cur in zip(labels, labels[1:]):
        if prev != cur:
            changes += 1
    return changes


class SessionMemory:

    def __init__(self, windows=(5, 10, 30)):
        self.windows = tuple(sorted(windows))
        self._max_window = max(self.windows)
        self._store = deque()
        self.trend_analyzer = SessionTrendAnalyzer()

    def reset(self):
        self._store.clear()


    def _snapshot(self, result):
        return {
            "emotion": result.get(
                "final_emotion",
                result.get("emotion", "unknown"),
            ),
            "stress_score": float(
                result.get("fused_stress_score", 0.0)
            ),
            "stress_level": result.get("stress_level", "LOW"),
            "blink_count": int(result.get("blink_count", 0)),
            "movement_score": float(
                result.get("movement_score", 0.0)
            ),
            "gaze_stability": float(
                result.get("gaze_stability", 1.0)
            ),
            "fusion_score": float(result.get("fusion_score", 0.0)),
            "speech_intensity": float(
                result.get("speech_intensity", 0.0)
            ),
            "cognitive_load": result.get("cognitive_load", "LOW"),
            "deception_risk": result.get("deception_risk", "LOW"),
        }

    def update(self, result, now=None):
        if now is None:
            now = time.time()

        self._store.append((now, self._snapshot(result)))

        cutoff = now - self._max_window
        while self._store and self._store[0][0] < cutoff:
            self._store.popleft()

        memory = {}
        for window in self.windows:
            memory[f"{window}s"] = self._window_stats(window, now)

        memory["current_vs_historical"] = (
            self._current_vs_historical(now)
        )

        return memory


    def _frames_in(self, window, now):
        lower = now - window
        return [
            snap for (t, snap) in self._store if t >= lower
        ]

    def _window_stats(self, window, now):
        frames = self._frames_in(window, now)

        if not frames:
            return {"frames": 0}

        emotions = [f["emotion"] for f in frames]
        stress_levels = [f["stress_level"] for f in frames]
        cognitive_loads = [f["cognitive_load"] for f in frames]
        deception_risks = [f["deception_risk"] for f in frames]
        stress_scores = [f["stress_score"] for f in frames]

        trend = self.trend_analyzer.analyze(
            emotions,
            stress_levels,
            cognitive_loads,
        )

        blink_start = frames[0]["blink_count"]
        blink_end = frames[-1]["blink_count"]
        elapsed = self._elapsed(window, now)
        blink_delta = max(blink_end - blink_start, 0)
        blink_frequency = (
            blink_delta / elapsed if elapsed > 0 else 0.0
        )

        return {
            "frames": len(frames),
            "dominant_emotion": trend["dominant_emotion"],
            "emotion_changes": _count_changes(emotions),
            "stress_trend": self._numeric_trend(stress_scores),
            "avg_stress_score": round(_mean(stress_scores), 3),
            "blink_frequency": round(blink_frequency, 3),
            "avg_head_movement": round(
                _mean([f["movement_score"] for f in frames]), 2
            ),
            "avg_gaze_stability": round(
                _mean([f["gaze_stability"] for f in frames]), 3
            ),
            "avg_fusion_score": round(
                _mean([f["fusion_score"] for f in frames]), 1
            ),
            "avg_speech_intensity": round(
                _mean([f["speech_intensity"] for f in frames]), 4
            ),
            "dominant_cognitive_load": _mode(cognitive_loads),
            "dominant_deception_risk": _mode(deception_risks),
        }

    def _elapsed(self, window, now):
        frames_t = [
            t for (t, _) in self._store if t >= now - window
        ]
        if len(frames_t) < 2:
            return 0.0
        return frames_t[-1] - frames_t[0]

    def _numeric_trend(self, scores):
        if len(scores) < 4:
            return "STABLE"
        mid = len(scores) // 2
        first = _mean(scores[:mid])
        second = _mean(scores[mid:])
        diff = second - first
        if diff > STRESS_TREND_THRESHOLD:
            return "INCREASING"
        if diff < -STRESS_TREND_THRESHOLD:
            return "DECREASING"
        return "STABLE"

    def _current_vs_historical(self, now):
        if not self._store:
            return {
                "delta": 0.0,
                "direction": "STABLE",
            }

        current = self._store[-1][1]["fusion_score"]
        historical = _mean(
            [f["fusion_score"] for f in self._frames_in(
                self._max_window, now
            )]
        )
        delta = current - historical

        if delta > FUSION_DELTA_THRESHOLD:
            direction = "RISING"
        elif delta < -FUSION_DELTA_THRESHOLD:
            direction = "FALLING"
        else:
            direction = "STABLE"

        return {
            "current_fusion_score": round(current, 1),
            "historical_fusion_score": round(historical, 1),
            "delta": round(delta, 1),
            "direction": direction,
        }
