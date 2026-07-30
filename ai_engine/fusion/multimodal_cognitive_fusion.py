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

STRESS_MAP = {
    "LOW": 0.0,
    "MEDIUM": 0.5,
    "HIGH": 1.0,
}

STRESS_HIGH_THRESHOLD = 0.66
STRESS_MEDIUM_THRESHOLD = 0.33

NEGATIVE_EMOTIONS = {"fear", "angry", "sad", "disgust"}
NEUTRAL_EMOTIONS = {"neutral"}

EMOTION_NEGATIVITY = {
    "negative": 1.0,
    "neutral": 0.2,
    "positive": 0.0,
}

BLINK_RATE_CEILING = 30.0
HEAD_MOVEMENT_CEILING = 20.0

WEIGHTS = {
    "visual_stress": 0.30,
    "audio_stress": 0.25,
    "emotion_negativity": 0.15,
    "blink": 0.10,
    "head_movement": 0.10,
    "eye_instability": 0.10,
}

COGNITIVE_LOAD_HIGH_THRESHOLD = 60.0
COGNITIVE_LOAD_MEDIUM_THRESHOLD = 35.0

DECEPTION_HIGH_THRESHOLD = 6
DECEPTION_MEDIUM_THRESHOLD = 3

EMOTION_AGREEMENT_BONUS = 0.10
EMOTION_DISAGREEMENT_PENALTY = 0.70

MODALITY_ABSENT_CONFIDENCE = 0.001


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def _stress_val(level):
    return STRESS_MAP.get(str(level).upper(), 0.0)


def _emotion_category(emotion):
    if emotion in NEGATIVE_EMOTIONS:
        return "negative"
    if emotion in NEUTRAL_EMOTIONS:
        return "neutral"
    return "positive"


class MultimodalCognitiveFusion:
    def __init__(self):
        self.temporal_engine = TemporalDecisionEngine()
        self.state_generator = CognitiveStateGenerator()
        self.risk_engine = RiskEngine()
        self.emotion_history = []

    def fuse_emotion(
        self,
        visual_emotion,
        visual_confidence,
        audio_emotion,
        audio_confidence,
    ):
        visual_present = (
            visual_confidence > MODALITY_ABSENT_CONFIDENCE
        )
        audio_present = (
            audio_confidence > MODALITY_ABSENT_CONFIDENCE
        )

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

    def fuse_stress(
        self,
        visual_stress,
        visual_confidence,
        audio_stress,
        audio_confidence,
    ):
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

    def compute_fusion_score(
        self,
        visual_stress_val,
        audio_stress_val,
        fused_emotion,
        blink_rate,
        head_movement,
        gaze_stability,
    ):
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

    def derive_cognitive_load(self, fusion_score):
        if fusion_score >= COGNITIVE_LOAD_HIGH_THRESHOLD:
            return "HIGH"
        if fusion_score >= COGNITIVE_LOAD_MEDIUM_THRESHOLD:
            return "MEDIUM"
        return "LOW"

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

        v_emo_conf = _clamp(visual_emotion_confidence * video_quality)
        a_emo_conf = _clamp(audio_emotion_confidence * audio_quality)
        v_str_conf = _clamp(visual_stress_confidence * video_quality)
        a_str_conf = _clamp(audio_stress_confidence * audio_quality)

        emotion = self.fuse_emotion(
            visual_emotion,
            v_emo_conf,
            audio_emotion,
            a_emo_conf,
        )
        final_emotion = emotion["emotion"]

        stress = self.fuse_stress(
            visual_stress,
            v_str_conf,
            audio_stress,
            a_str_conf,
        )

        _wsum = v_emo_conf + a_emo_conf
        modality_weights = {
            "visual": round(v_emo_conf / _wsum, 3) if _wsum else 0.5,
            "audio": round(a_emo_conf / _wsum, 3) if _wsum else 0.5,
        }

        quality_warning = (
            max(video_quality, audio_quality) < QUALITY_WARNING_THRESHOLD
        )

        fusion_score = self.compute_fusion_score(
            visual_stress_val=_stress_val(visual_stress),
            audio_stress_val=_stress_val(audio_stress),
            fused_emotion=final_emotion,
            blink_rate=blink_rate,
            head_movement=head_movement,
            gaze_stability=gaze_stability,
        )

        cognitive_load = self.derive_cognitive_load(fusion_score)

        deception_risk = self.derive_deception_risk(
            cognitive_load,
            fusion_score,
            gaze_shift_frequency,
            final_emotion,
        )

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
