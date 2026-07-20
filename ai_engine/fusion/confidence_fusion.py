class ConfidenceFusion:

    def weighted_emotion_fusion(

        self,

        visual_emotion,
        visual_confidence,

        audio_emotion,
        audio_confidence

    ):

        scores = {}

        scores[visual_emotion] = (
            scores.get(
                visual_emotion,
                0
            )
            + visual_confidence
        )

        scores[audio_emotion] = (
            scores.get(
                audio_emotion,
                0
            )
            + audio_confidence
        )

        final_emotion = max(
            scores,
            key=scores.get
        )

        confidence = (
            scores[final_emotion] / 2
        )

        return {

            "emotion":
                final_emotion,

            "confidence":
                confidence
        }