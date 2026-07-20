class CognitiveExplainer:

    def explain(

        self,

        blink_rate,

        gaze_stability,

        movement_score,

        cognitive_load,

        deception_risk

    ):

        reasons = []

        if blink_rate > 20:

            reasons.append(
                "High blink rate detected"
            )

        if gaze_stability < 0.6:

            reasons.append(
                "Unstable gaze pattern"
            )

        if movement_score > 5:

            reasons.append(
                "Frequent head movement"
            )

        if cognitive_load == "HIGH":

            reasons.append(
                "High cognitive load"
            )

        if deception_risk == "HIGH":

            reasons.append(
                "Potential deception indicators"
            )

        if len(reasons) == 0:

            reasons.append(
                "Normal behavioral activity"
            )

        return reasons