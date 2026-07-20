class FinalCognitiveRisk:

    def calculate(

        self,

        stress_level,

        cognitive_load,

        deception_risk,

        anomaly_risk

    ):

        score = 0

        mappings = {
            "LOW":1,
            "MEDIUM":2,
            "HIGH":3
        }

        score += mappings.get(
            stress_level,
            1
        )

        score += mappings.get(
            cognitive_load,
            1
        )

        score += mappings.get(
            deception_risk,
            1
        )

        score += mappings.get(
            anomaly_risk,
            1
        )

        if score >= 10:
            risk = "CRITICAL"

        elif score >= 7:
            risk = "HIGH"

        elif score >= 4:
            risk = "MEDIUM"

        else:
            risk = "LOW"

        return {
            "score": score,
            "risk": risk
        }