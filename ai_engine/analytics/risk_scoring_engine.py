class RiskScoringEngine:

    def calculate(

        self,

        stress_level,
        cognitive_load,
        deception_risk,
        anomaly_risk
    ):

        score = 0

        if stress_level == "HIGH":
            score += 30

        elif stress_level == "MEDIUM":
            score += 15

        if cognitive_load == "HIGH":
            score += 25

        elif cognitive_load == "MEDIUM":
            score += 10

        if deception_risk == "HIGH":
            score += 30

        elif deception_risk == "MEDIUM":
            score += 15

        if anomaly_risk == "HIGH":
            score += 15

        elif anomaly_risk == "MEDIUM":
            score += 8

        if score >= 70:
            level = "CRITICAL"

        elif score >= 40:
            level = "ELEVATED"

        else:
            level = "NORMAL"

        return {

            "score": score,
            "level": level
        }
