class AnomalyDetector:

    def detect(

        self,

        gaze_stability,
        movement_score,
        blink_rate,
        stress_level

    ):

        anomaly_score = 0

        if gaze_stability < 0.4:
            anomaly_score += 1

        if movement_score > 15:
            anomaly_score += 1

        if blink_rate > 40:
            anomaly_score += 1

        if stress_level == "HIGH":
            anomaly_score += 1

        if anomaly_score >= 3:

            return {
                "anomaly": True,
                "risk": "HIGH"
            }

        elif anomaly_score == 2:

            return {
                "anomaly": True,
                "risk": "MEDIUM"
            }

        return {
            "anomaly": False,
            "risk": "LOW"
        }