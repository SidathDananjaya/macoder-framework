from ai_engine.fusion.confidence_fusion import ConfidenceFusion

from ai_engine.fusion.temporal_decision_engine import (
    TemporalDecisionEngine
)

from ai_engine.fusion.cognitive_state_generator import (
    CognitiveStateGenerator
)


class MultimodalFusionEngine:

    def __init__(self):

        self.confidence_fusion = (
            ConfidenceFusion()
        )

        self.temporal_engine = (
            TemporalDecisionEngine()
        )

        self.state_generator = (
            CognitiveStateGenerator()
        )

        self.emotion_history = []

    def process(
        self,
        visual_emotion,
        visual_confidence,
        temporal_emotion,
        temporal_confidence,
        stress_level,
        cognitive_load,
        deception_risk
    ):

        fusion = (
            self.confidence_fusion
            .weighted_emotion_fusion(
                visual_emotion,
                visual_confidence,
                temporal_emotion,
                temporal_confidence
            )
        )

        final_emotion = fusion["emotion"]

        self.emotion_history.append(
            final_emotion
        )

        temporal_state = (
            self.temporal_engine.analyze(
                self.emotion_history
            )
        )

        cognitive_state = (
            self.state_generator.generate(
                final_emotion,
                stress_level,
                cognitive_load,
                deception_risk,
                temporal_state
            )
        )

        return {

            "final_emotion": final_emotion,

            "fusion_confidence":
                fusion["confidence"],

            "temporal_state":
                temporal_state,

            "cognitive_state":
                cognitive_state
        }