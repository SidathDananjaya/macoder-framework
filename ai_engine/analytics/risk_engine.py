"""
Phase 4 - Risk Engine.

Replaces the coarse categorical verdict ("Risk = LOW") with a quantitative
**Risk Score 0-100** produced by a transparent, weighted rule set:

    Visual Stress        35%
    Audio Stress         30%
    Emotion Instability  15%
    Eye Avoidance        10%
    Head Movement        10%
                    |
                    v
            Risk Score (0 - 100)  ->  Risk Level (LOW / MEDIUM / HIGH)

This is the Phase 4 authority for the numeric risk metric. The older scorers
``FinalCognitiveRisk`` and ``RiskScoringEngine`` are earlier drafts that are not
on the live pipeline and should be considered superseded by this class.

Every weight and threshold is a named constant so the score can be explained and
defended in the dissertation. Each ``compute`` call also returns a
``risk_breakdown`` giving the per-signal contribution to the total, so the
dashboard (and the later LLM report phase) can explain *why* the score is what it
is.

The constants below intentionally mirror the fusion engine's own normalisation
idioms (``STRESS_MAP``, ``HEAD_MOVEMENT_CEILING``). They are redeclared locally
rather than imported so that ``multimodal_cognitive_fusion`` can depend on this
module without a circular import.
"""


# Categorical stress -> numeric intensity (0..1). Mirrors the fusion engine.
STRESS_MAP = {
    "LOW": 0.0,
    "MEDIUM": 0.5,
    "HIGH": 1.0,
}

# Normalisation ceiling for head movement (value that maps to 1.0).
HEAD_MOVEMENT_CEILING = 20.0

# Number of most-recent emotion labels used to gauge emotion instability.
EMOTION_WINDOW = 30

# Risk score component weights (must sum to 1.0).
RISK_WEIGHTS = {
    "visual_stress": 0.35,
    "audio_stress": 0.30,
    "emotion_instability": 0.15,
    "eye_avoidance": 0.10,
    "head_movement": 0.10,
}

# Risk score (0..100) -> risk level.
RISK_HIGH_THRESHOLD = 70.0    # >= 70 -> HIGH   (red)
RISK_MEDIUM_THRESHOLD = 40.0  # 40-69 -> MEDIUM (amber); else LOW (green)

# Exponential-moving-average smoothing on the final 0..100 score. Higher =
# more responsive, lower = smoother. 0.4 keeps the gauge lively but not jittery.
EMA_ALPHA = 0.4


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def _count_changes(labels):
    """Number of times a label differs from the previous one."""
    changes = 0
    for prev, cur in zip(labels, labels[1:]):
        if prev != cur:
            changes += 1
    return changes


def _emotion_instability(emotion_history):
    """Frame-to-frame emotion change-rate over the recent window (0..1).

    Uses the last ``EMOTION_WINDOW`` labels. The rate is the number of label
    changes divided by the number of adjacent pairs, so a perfectly steady
    emotion scores 0.0 and an emotion that changes every frame scores 1.0.
    """
    recent = list(emotion_history)[-EMOTION_WINDOW:]
    if len(recent) < 2:
        return 0.0
    return _clamp(_count_changes(recent) / (len(recent) - 1))


class RiskEngine:
    """Weighted, explainable Risk Score (0-100) engine.

    Stateful only in that it holds the previous smoothed score for the EMA.
    Instantiate once and reuse across frames of a session; call ``reset`` when a
    new session begins.
    """

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
        """Compute the weighted risk score for the current frame.

        Parameters
        ----------
        visual_stress, audio_stress : str
            Categorical stress levels ("LOW"/"MEDIUM"/"HIGH").
        emotion_history : sequence of str
            Chronological emotion labels; the most recent window is used to
            gauge emotion instability.
        head_movement : float
            Raw head-movement score (normalised by ``HEAD_MOVEMENT_CEILING``).
        gaze_stability : float
            Gaze stability 0..1 (1.0 = perfectly steady). Eye avoidance is its
            complement.

        Returns
        -------
        dict with keys ``risk_score`` (float 0-100, EMA-smoothed),
        ``risk_level`` (str) and ``risk_breakdown`` (per-signal contribution in
        0-100 points, summing to the raw score before smoothing).
        """

        components = {
            "visual_stress": STRESS_MAP.get(visual_stress, 0.0),
            "audio_stress": STRESS_MAP.get(audio_stress, 0.0),
            "emotion_instability": _emotion_instability(emotion_history),
            "eye_avoidance": _clamp(1.0 - gaze_stability),
            "head_movement": _clamp(
                head_movement / HEAD_MOVEMENT_CEILING
            ),
        }

        # Per-signal contribution in 0..100 points (weight x component x 100).
        breakdown = {
            name: round(RISK_WEIGHTS[name] * value * 100, 1)
            for name, value in components.items()
        }

        raw_score = _clamp(sum(breakdown.values()), 0.0, 100.0)

        # Exponential moving average to de-jitter the live gauge.
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
