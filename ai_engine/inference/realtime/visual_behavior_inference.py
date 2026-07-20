import pandas as pd
import joblib


class VisualBehaviorInference:

    def __init__(self):

        self.model = joblib.load(
            "experiments/exp_005_visual_behavior/"
            "visual_behavior_model.pkl"
        )

        self.encoder = joblib.load(
            "experiments/exp_005_visual_behavior/"
            "visual_behavior_label_encoder.pkl"
        )

        self.scaler = joblib.load(
            "experiments/exp_005_visual_behavior/"
            "visual_scaler.pkl"
        )

    def predict(self, features_dict):

        df = pd.DataFrame([features_dict])

        prediction = self.model.predict(df)[0]

        probabilities = self.model.predict_proba(df)[0]

        confidence = max(probabilities)

        emotion = self.encoder.inverse_transform(
            [prediction]
        )[0]

        scaled_features = self.scaler.transform(df)

        prediction = self.model.predict(
            scaled_features
        )[0]

        probabilities = self.model.predict_proba(
            scaled_features
        )[0]

        return {
            "emotion": emotion,
            "confidence": float(confidence)
        }