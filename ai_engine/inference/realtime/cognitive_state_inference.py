import joblib
import pandas as pd

class CognitiveStateInference:

    def __init__(self):

        self.model = joblib.load(
            "experiments/exp_008_cognitive/"
            "cognitive_model.pkl"
        )

        self.encoder = joblib.load(
            "experiments/exp_008_cognitive/"
            "label_encoder.pkl"
        )

        self.scaler = joblib.load(
            "experiments/exp_008_cognitive/"
            "scaler.pkl"
        )

    def predict(self, features):

        df = pd.DataFrame([features])

        scaled = self.scaler.transform(df)

        prediction = self.model.predict(scaled)[0]

        probabilities = self.model.predict_proba(
            scaled
        )[0]

        confidence = max(probabilities)

        state = self.encoder.inverse_transform(
            [prediction]
        )[0]

        return {
            "state": state,
            "confidence": float(confidence)
        }