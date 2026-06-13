import joblib
import pandas as pd


class RealtimeVisualPredictor:

    def __init__(self):

        self.model = joblib.load(
            "experiments/exp_002_visual_baseline/visual_behavior_model.pkl"
        )

        self.scaler = joblib.load(
            "experiments/exp_002_visual_baseline/visual_scaler.pkl"
        )

        self.label_encoder = joblib.load(
            "experiments/exp_002_visual_baseline/visual_label_encoder.pkl"
        )

    def predict(
        self,
        features_dict
    ):

        df = pd.DataFrame([features_dict])

        scaled = self.scaler.transform(df)

        prediction = self.model.predict(scaled)

        label = self.label_encoder.inverse_transform(
            prediction
        )[0]

        return label