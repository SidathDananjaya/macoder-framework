class ConfidenceFusion:

    def calculate(

        self,

        visual_confidence,
        temporal_confidence,
        anomaly_risk

    ):

        confidence = (
            visual_confidence * 0.4 +
            temporal_confidence * 0.6
        )

        if anomaly_risk == "HIGH":

            confidence *= 0.85

        elif anomaly_risk == "MEDIUM":

            confidence *= 0.93

        return round(confidence, 2)