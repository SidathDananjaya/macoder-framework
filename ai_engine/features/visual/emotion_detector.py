from deepface import DeepFace


class EmotionDetector:

    def __init__(self):
        pass

    def analyze_emotion(self, frame):

        try:

            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )

            if isinstance(result, list):
                result = result[0]

            dominant_emotion = result["dominant_emotion"]

            emotion_scores = result["emotion"]

            return {
                "dominant_emotion": dominant_emotion,
                "emotion_scores": emotion_scores
            }

        except Exception as e:

            print("Emotion detection error:", e)

            return {
                "dominant_emotion": "unknown",
                "emotion_scores": {}
            }