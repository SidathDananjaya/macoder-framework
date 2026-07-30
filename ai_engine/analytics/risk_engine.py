
STRESS_MAP = {
    "LOW": 0.0,
    "MEDIUM": 0.5,
    "HIGH": 1.0,
}

HEAD_MOVEMENT_CEILING = 20.0

EMOTION_WINDOW = 30

RISK_WEIGHTS = {
    "visual_stress": 0.35,
    "audio_stress": 0.30,
    "emotion_instability": 0.15,
    "eye_avoidance": 0.10,
    "head_movement": 0.10,
}

RISK_HIGH_THRESHOLD = 70.0
RISK_MEDIUM_THRESHOLD = 40.0

EMA_ALPHA = 0.4


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def _count_changes(labels):
    changes = 0
    for prev, cur in zip(labels, labels[1:]):
        if prev != cur:
            changes += 1
    return changes


def _emotion_instability(emotion_history):
    recent = list(emotion_history)[-EMOTION_WINDOW:]
    if len(recent) < 2:
        return 0.0
    return _clamp(_count_changes(recent) / (len(recent) - 1))


class RiskEngine:
    def __init__(self):
        self._prev_score = None

    def reset(self):
        self._prev_score = None

    def compute(
        self,
        visual_stress,
        audio_stress,
        emotion_history,
        head_movement,
        gaze_stability,
    ):
        components = {
            "visual_stress": STRESS_MAP.get(visual_stress, 0.0),
            "audio_stress": STRESS_MAP.get(audio_stress, 0.0),
            "emotion_instability": _emotion_instability(emotion_history),
            "eye_avoidance": _clamp(1.0 - gaze_stability),
            "head_movement": _clamp(
                head_movement / HEAD_MOVEMENT_CEILING
            ),
        }

        breakdown = {
            name: round(RISK_WEIGHTS[name] * value * 100, 1)
            for name, value in components.items()
        }

        raw_score = _clamp(sum(breakdown.values()), 0.0, 100.0)

        if self._prev_score is None:
            smoothed = raw_score
        else:
            smoothed = (
                EMA_ALPHA * raw_score
                + (1 - EMA_ALPHA) * self._prev_score
            )
        self._prev_score = smoothed

        risk_score = round(smoothed, 1)

        return {
            "risk_score": risk_score,
            "risk_level": self._risk_level(risk_score),
            "risk_breakdown": breakdown,
        }

    def _risk_level(self, risk_score):
        if risk_score >= RISK_HIGH_THRESHOLD:
            return "HIGH"
        if risk_score >= RISK_MEDIUM_THRESHOLD:
            return "MEDIUM"
        return "LOW"
