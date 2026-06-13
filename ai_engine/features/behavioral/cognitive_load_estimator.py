class CognitiveLoadEstimator:

    def estimate(
        self,
        blink_rate,
        emotion,
        gaze_stability,
        movement_score
    ):

        score = 0

        # Blink rate contribution
        if blink_rate > 25:
            score += 2
        elif blink_rate > 15:
            score += 1

        # Emotion contribution
        if emotion in ["fear", "angry", "sad"]:
            score += 2

        # Gaze instability
        if gaze_stability < 0.5:
            score += 2
        elif gaze_stability < 0.7:
            score += 1

        # Excessive movement
        if movement_score > 15:
            score += 2
        elif movement_score > 8:
            score += 1

        # Final level
        if score >= 6:
            return "HIGH"

        elif score >= 3:
            return "MEDIUM"

        return "LOW"