class CognitiveStateGenerator:

    def generate(
        self,
        emotion,
        stress,
        cognitive_load,
        deception_risk,
        temporal_state
    ):

        if (
            stress == "HIGH"
            and cognitive_load == "HIGH"
        ):

            return (
                "High cognitive stress detected"
            )

        if (
            deception_risk == "HIGH"
            and temporal_state == "UNSTABLE"
        ):

            return (
                "Possible deceptive behavior"
            )

        if emotion in ["fear", "angry"]:
            return (
                "Negative emotional state"
            )

        return "Stable emotional condition"