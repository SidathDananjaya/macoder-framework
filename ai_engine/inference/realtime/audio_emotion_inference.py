import joblib
import pandas as pd


class AudioEmotionInference:

    def __init__(self):

        self.model = joblib.load(
            "experiments/exp_009_audio_emotion_v2/"
            "audio_emotion_model.pkl"
        )

        self.encoder = joblib.load(
            "experiments/exp_009_audio_emotion_v2/"
            "audio_label_encoder.pkl"
        )

        self.scaler = joblib.load(
            "experiments/exp_009_audio_emotion_v2/"
            "audio_scaler.pkl"
        )

    def predict(self, features):

        df = pd.DataFrame([features])

        scaled = self.scaler.transform(df)

        prediction = self.model.predict(
            scaled
        )[0]

        probabilities = self.model.predict_proba(
            scaled
        )[0]

        confidence = max(probabilities)

        emotion = self.encoder.inverse_transform(
            [prediction]
        )[0]

        return {

            "emotion": emotion,

            "confidence": float(
                confidence
            )
        }