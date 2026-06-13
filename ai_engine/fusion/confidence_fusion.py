class ConfidenceFusion:

    def weighted_emotion_fusion(
        self,
        visual_emotion,
        visual_confidence,
        temporal_emotion,
        temporal_confidence
    ):

        scores = {}

        # Visual score
        scores[visual_emotion] = (
            scores.get(visual_emotion, 0)
            + visual_confidence
        )

        # Temporal score
        scores[temporal_emotion] = (
            scores.get(temporal_emotion, 0)
            + temporal_confidence
        )

        final_emotion = max(
            scores,
            key=scores.get
        )

        confidence = scores[final_emotion] / 2

        return {
            "emotion": final_emotion,
            "confidence": confidence
        }