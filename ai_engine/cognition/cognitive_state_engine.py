class CognitiveStateEngine:

    def evaluate(

        self,

        stress_level,
        cognitive_load,
        deception_risk,
        anomaly_risk,
        temporal_emotion

    ):

        if (
            stress_level == "HIGH"
            and deception_risk == "HIGH"
        ):

            return (
                "Potential deceptive cognitive state"
            )

        if (
            cognitive_load == "HIGH"
            and anomaly_risk == "HIGH"
        ):

            return (
                "Elevated cognitive stress detected"
            )

        if temporal_emotion in [
            "fear",
            "sad",
            "disgust"
        ]:

            return (
                "Negative emotional trajectory"
            )

        return (
            "Stable emotional condition"
        )