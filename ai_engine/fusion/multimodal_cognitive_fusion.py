"""
Phase 2 - Proper Multimodal Fusion.

Implements the dissertation fusion chain:

    Visual Emotion + Audio Emotion       -> fused emotion
    Visual Stress  + Audio Stress        -> fused stress
    Blink Rate + Head Movement + Eye Stability
                    |
                    v
            Fusion Score (0 - 100)
                    |
                    v
            Cognitive Load (LOW / MEDIUM / HIGH)
                    |
                    v
            Deception Risk (LOW / MEDIUM / HIGH)

All logic is deterministic and rule based so it can be explained and defended
in the dissertation. Every weight and threshold is declared as a named
constant below.
"""

from ai_engine.fusion.temporal_decision_engine import (
    TemporalDecisionEngine
)

from ai_engine.fusion.cognitive_state_generator import (
    CognitiveStateGenerator
)

from ai_engine.analytics.risk_engine import (
    RiskEngine
)

from ai_engine.features.quality.quality_assessor import (
    QUALITY_WARNING_THRESHOLD
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Categorical stress -> numeric intensity (0..1)
STRESS_MAP = {
    "LOW": 0.0,
    "MEDIUM": 0.5,
    "HIGH": 1.0,
}

# Reverse mapping thresholds for numeric intensity -> categorical stress
STRESS_HIGH_THRESHOLD = 0.66
STRESS_MEDIUM_THRESHOLD = 0.33

# Emotions that indicate negative / high-arousal cognitive state
NEGATIVE_EMOTIONS = {"fear", "angry", "sad", "disgust"}
NEUTRAL_EMOTIONS = {"neutral"}

# Emotion -> negativity contribution (0..1) for the fusion score
EMOTION_NEGATIVITY = {
    "negative": 1.0,
    "neutral": 0.2,
    "positive": 0.0,
}

# Normalisation ceilings for behavioural signals (value that maps to 1.0)
BLINK_RATE_CEILING = 30.0
HEAD_MOVEMENT_CEILING = 20.0

# Fusion score component weights (must sum to 1.0)
WEIGHTS = {
    "visual_stress": 0.30,
    "audio_stress": 0.25,
    "emotion_negativity": 0.15,
    "blink": 0.10,
    "head_movement": 0.10,
    "eye_instability": 0.10,
}

# Fusion score (0..100) -> cognitive load
COGNITIVE_LOAD_HIGH_THRESHOLD = 60.0
COGNITIVE_LOAD_MEDIUM_THRESHOLD = 35.0

# Deception risk scoring thresholds
DECEPTION_HIGH_THRESHOLD = 6
DECEPTION_MEDIUM_THRESHOLD = 3

# Confidence adjustments for emotion fusion
EMOTION_AGREEMENT_BONUS = 0.10
EMOTION_DISAGREEMENT_PENALTY = 0.70

# A confidence at or below this is treated as "modality absent"
MODALITY_ABSENT_CONFIDENCE = 0.001


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def _stress_val(level):
    """Categorical stress -> numeric intensity, case-insensitively.

    The visual stress arrives upper case but the audio stress model returns
    lower case ("medium"/"high"); normalising here means an audio "medium" is
    no longer silently treated as 0.0 (LOW) in the fusion.
    """
    return STRESS_MAP.get(str(level).upper(), 0.0)


def _emotion_category(emotion):
    if emotion in NEGATIVE_EMOTIONS:
        return "negative"
    if emotion in NEUTRAL_EMOTIONS:
        return "neutral"
    return "positive"


class MultimodalCognitiveFusion:
    """Fuses visual and audio signals into cognitive-state estimates."""

    def __init__(self):
        self.temporal_engine = TemporalDecisionEngine()
        self.state_generator = CognitiveStateGenerator()
        self.risk_engine = RiskEngine()
        self.emotion_history = []

    # ------------------------------------------------------------------
    # Emotion fusion
    # ------------------------------------------------------------------

    def fuse_emotion(
        self,
        visual_emotion,
        visual_confidence,
        audio_emotion,
        audio_confidence,
    ):
        """Combine visual and audio emotion into a single decision.

        Returns dict: {emotion, confidence, agreement}
        agreement is True/False when both modalities are present, else None.
        """

        visual_present = (
            visual_confidence > MODALITY_ABSENT_CONFIDENCE
        )
        audio_present = (
            audio_confidence > MODALITY_ABSENT_CONFIDENCE
        )

        # Single modality (or none) available.
        if visual_present and not audio_present:
            return {
                "emotion": visual_emotion,
                "confidence": _clamp(visual_confidence),
                "agreement": None,
            }

        if audio_present and not visual_present:
            return {
                "emotion": audio_emotion,
                "confidence": _clamp(audio_confidence),
                "agreement": None,
            }

        if not visual_present and not audio_present:
            return {
                "emotion": visual_emotion,
                "confidence": 0.0,
                "agreement": None,
            }

        # Both modalities present.
        if visual_emotion == audio_emotion:
            confidence = _clamp(
                (visual_confidence + audio_confidence) / 2
                + EMOTION_AGREEMENT_BONUS
            )
            return {
                "emotion": visual_emotion,
                "confidence": confidence,
                "agreement": True,
            }

        # Disagreement -> trust the more confident modality, penalise.
        if visual_confidence >= audio_confidence:
            winner, winner_conf = visual_emotion, visual_confidence
        else:
            winner, winner_conf = audio_emotion, audio_confidence

        return {
            "emotion": winner,
            "confidence": _clamp(
                winner_conf * EMOTION_DISAGREEMENT_PENALTY
            ),
            "agreement": False,
        }

    # ------------------------------------------------------------------
    # Stress fusion
    # ------------------------------------------------------------------

    def fuse_stress(
        self,
        visual_stress,
        visual_confidence,
        audio_stress,
        audio_confidence,
    ):
        """Confidence-weighted fusion of categorical stress levels.

        Returns dict: {level, score} where score is the numeric intensity.
        """

        visual_val = _stress_val(visual_stress)
        audio_val = _stress_val(audio_stress)

        visual_w = (
            visual_confidence
            if visual_confidence > MODALITY_ABSENT_CONFIDENCE
            else 0.0
        )
        audio_w = (
            audio_confidence
            if audio_confidence > MODALITY_ABSENT_CONFIDENCE
            else 0.0
        )

        if visual_w + audio_w == 0:
            # No usable confidence -> fall back to equal weighting.
            fused_val = (visual_val + audio_val) / 2
        else:
            fused_val = (
                visual_val * visual_w + audio_val * audio_w
            ) / (visual_w + audio_w)

        if fused_val >= STRESS_HIGH_THRESHOLD:
            level = "HIGH"
        elif fused_val >= STRESS_MEDIUM_THRESHOLD:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {"level": level, "score": round(fused_val, 3)}

    # ------------------------------------------------------------------
    # Fusion score
    # ------------------------------------------------------------------

    def compute_fusion_score(
        self,
        visual_stress_val,
        audio_stress_val,
        fused_emotion,
        blink_rate,
        head_movement,
        gaze_stability,
    ):
        """Weighted 0..100 fusion score across all modalities."""

        emotion_component = EMOTION_NEGATIVITY[
            _emotion_category(fused_emotion)
        ]

        blink_component = _clamp(blink_rate / BLINK_RATE_CEILING)
        movement_component = _clamp(
            head_movement / HEAD_MOVEMENT_CEILING
        )
        eye_instability_component = _clamp(1.0 - gaze_stability)

        score = (
            WEIGHTS["visual_stress"] * visual_stress_val
            + WEIGHTS["audio_stress"] * audio_stress_val
            + WEIGHTS["emotion_negativity"] * emotion_component
            + WEIGHTS["blink"] * blink_component
            + WEIGHTS["head_movement"] * movement_component
            + WEIGHTS["eye_instability"] * eye_instability_component
        )

        return round(_clamp(score) * 100, 1)

    # ------------------------------------------------------------------
    # Cognitive load
    # ------------------------------------------------------------------

    def derive_cognitive_load(self, fusion_score):
        if fusion_score >= COGNITIVE_LOAD_HIGH_THRESHOLD:
            return "HIGH"
        if fusion_score >= COGNITIVE_LOAD_MEDIUM_THRESHOLD:
            return "MEDIUM"
        return "LOW"

    # ------------------------------------------------------------------
    # Deception risk
    # ------------------------------------------------------------------

    def derive_deception_risk(
        self,
        cognitive_load,
        fusion_score,
        gaze_shift_frequency,
        fused_emotion,
    ):
        points = 0

        if cognitive_load == "HIGH":
            points += 3
        elif cognitive_load == "MEDIUM":
            points += 1

        if fusion_score >= 65:
            points += 2
        elif fusion_score >= 45:
            points += 1

        if gaze_shift_frequency > 20:
            points += 2
        elif gaze_shift_frequency > 10:
            points += 1

        if fused_emotion in {"fear", "angry"}:
            points += 1

        if points >= DECEPTION_HIGH_THRESHOLD:
            return "HIGH"
        if points >= DECEPTION_MEDIUM_THRESHOLD:
            return "MEDIUM"
        return "LOW"

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def process(
        self,
        visual_emotion,
        visual_emotion_confidence,
        audio_emotion,
        audio_emotion_confidence,
        visual_stress,
        visual_stress_confidence,
        audio_stress,
        audio_stress_confidence,
        blink_rate,
        head_movement,
        gaze_stability,
        gaze_shift_frequency,
        video_quality=1.0,
        audio_quality=1.0,
    ):
        """Run the full fusion chain and return the combined cognitive state.

        ``video_quality`` / ``audio_quality`` are the real-time signal-quality
        scores in [0, 1] (Section 5.4.2). They make the fusion **quality-adaptive**
        (Section 5.4.4): each modality's confidence is scaled by its channel quality
        before fusion, so a degraded channel (e.g. a dark frame or silent audio) is
        automatically down-weighted and, when quality collapses, dropped in favour of
        the more reliable modality.
        """

        # Quality-adaptive weighting: scale each modality's confidence by its
        # channel quality so poor lighting / noisy or absent audio reduces that
        # modality's influence (graceful degradation).
        v_emo_conf = _clamp(visual_emotion_confidence * video_quality)
        a_emo_conf = _clamp(audio_emotion_confidence * audio_quality)
        v_str_conf = _clamp(visual_stress_confidence * video_quality)
        a_str_conf = _clamp(audio_stress_confidence * audio_quality)

        # 1. Fuse emotion (quality-weighted confidences).
        emotion = self.fuse_emotion(
            visual_emotion,
            v_emo_conf,
            audio_emotion,
            a_emo_conf,
        )
        final_emotion = emotion["emotion"]

        # 2. Fuse stress (quality-weighted confidences).
        stress = self.fuse_stress(
            visual_stress,
            v_str_conf,
            audio_stress,
            a_str_conf,
        )

        # Dynamic modality weights actually applied (for the dashboard/report).
        _wsum = v_emo_conf + a_emo_conf
        modality_weights = {
            "visual": round(v_emo_conf / _wsum, 3) if _wsum else 0.5,
            "audio": round(a_emo_conf / _wsum, 3) if _wsum else 0.5,
        }

        # Warn only when NO modality provides a reliable signal (the best available
        # channel is below threshold). This refines Code Listing 2's per-modality
        # "any < 0.4" rule so that legitimate single-modality operation - e.g.
        # running on good video while audio is simply not connected - is not
        # constantly flagged as an error.
        quality_warning = (
            max(video_quality, audio_quality) < QUALITY_WARNING_THRESHOLD
        )

        # 3. Fusion score across all modalities.
        fusion_score = self.compute_fusion_score(
            visual_stress_val=_stress_val(visual_stress),
            audio_stress_val=_stress_val(audio_stress),
            fused_emotion=final_emotion,
            blink_rate=blink_rate,
            head_movement=head_movement,
            gaze_stability=gaze_stability,
        )

        # 4. Cognitive load from the fusion score.
        cognitive_load = self.derive_cognitive_load(fusion_score)

        # 5. Deception risk from cognitive load + supporting signals.
        deception_risk = self.derive_deception_risk(
            cognitive_load,
            fusion_score,
            gaze_shift_frequency,
            final_emotion,
        )

        # 6. Temporal + descriptive cognitive state.
        self.emotion_history.append(final_emotion)
        temporal_state = self.temporal_engine.analyze(
            self.emotion_history
        )

        cognitive_state = self.state_generator.generate(
            final_emotion,
            stress["level"],
            cognitive_load,
            deception_risk,
            temporal_state,
        )

        # 7. Phase 4 - quantitative weighted risk score (0..100). Uses the
        # per-frame categorical stress values plus the recent emotion history
        # (already updated above so the current frame is included).
        risk = self.risk_engine.compute(
            visual_stress=visual_stress,
            audio_stress=audio_stress,
            emotion_history=self.emotion_history,
            head_movement=head_movement,
            gaze_stability=gaze_stability,
        )

        return {
            "final_emotion": final_emotion,
            "emotion_agreement": emotion["agreement"],
            "fusion_confidence": round(emotion["confidence"], 3),
            "fused_stress": stress["level"],
            "fused_stress_score": stress["score"],
            "fusion_score": fusion_score,
            "cognitive_load": cognitive_load,
            "deception_risk": deception_risk,
            "risk_score": risk["risk_score"],
            "risk_level": risk["risk_level"],
            "risk_breakdown": risk["risk_breakdown"],
            "temporal_state": temporal_state,
            "cognitive_state": cognitive_state,
            "modality_weights": modality_weights,
            "quality_warning": quality_warning,
        }
