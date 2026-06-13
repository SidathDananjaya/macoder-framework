class TemporalDecisionEngine:

    def analyze(
        self,
        emotion_history
    ):

        if len(emotion_history) < 10:
            return "STABLE"

        unique_emotions = len(
            set(emotion_history[-10:])
        )

        if unique_emotions >= 5:
            return "UNSTABLE"

        elif unique_emotions >= 3:
            return "FLUCTUATING"

        return "STABLE"