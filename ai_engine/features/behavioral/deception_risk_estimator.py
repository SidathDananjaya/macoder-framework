class DeceptionRiskEstimator:

    def estimate(
        self,
        gaze_shift_frequency,
        movement_score,
        cognitive_load,
        emotion
    ):

        risk = 0

        if gaze_shift_frequency > 20:
            risk += 2

        if movement_score > 15:
            risk += 2

        if cognitive_load == "HIGH":
            risk += 2

        if emotion in ["fear", "angry"]:
            risk += 1

        if risk >= 6:
            return "HIGH"

        elif risk >= 3:
            return "MEDIUM"

        return "LOW"