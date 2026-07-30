class CognitiveFusionEngine:

    def process(

        self,

        visual_emotion,
        visual_confidence,

        temporal_emotion,
        temporal_confidence,

        stress_level,

        cognitive_load,

        deception_risk,

        anomaly_risk

    ):

        confidence = (
            visual_confidence +
            temporal_confidence
        ) / 2

        score = 0

        if stress_level == "HIGH":
            score += 3

        elif stress_level == "MEDIUM":
            score += 2

        if cognitive_load == "HIGH":
            score += 3

        elif cognitive_load == "MEDIUM":
            score += 2

        if deception_risk == "HIGH":
            score += 3

        elif deception_risk == "MEDIUM":
            score += 2

        if anomaly_risk == "HIGH":
            score += 3

        elif anomaly_risk == "MEDIUM":
            score += 2


        if score >= 10:

            cognitive_state = (
                "HIGH_COGNITIVE_STRESS"
            )

        elif score >= 6:

            cognitive_state = (
                "ELEVATED_COGNITIVE_EFFORT"
            )

        else:

            cognitive_state = (
                "STABLE_COGNITIVE_STATE"
            )

        return {

            "emotion":
                temporal_emotion,

            "confidence":
                confidence,

            "cognitive_state":
                cognitive_state,

            "score":
                score
        }
