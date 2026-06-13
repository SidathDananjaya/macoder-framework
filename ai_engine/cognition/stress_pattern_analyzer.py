class StressPatternAnalyzer:

    def analyze(self, stress_values):

        if len(stress_values) < 10:

            return "INSUFFICIENT_DATA"

        avg = sum(stress_values) / len(stress_values)

        recent = stress_values[-1]

        if recent > avg + 0.25:

            return "RISING_STRESS"

        elif recent < avg - 0.25:

            return "REDUCING_STRESS"

        else:

            return "STABLE_STRESS"