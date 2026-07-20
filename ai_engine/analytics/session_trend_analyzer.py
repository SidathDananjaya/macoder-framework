from collections import Counter


class SessionTrendAnalyzer:

    def analyze(

        self,

        emotions,

        stress_levels,

        cognitive_states

    ):

        if len(emotions) == 0:

            return {

                "dominant_emotion":
                    "unknown",

                "stress_trend":
                    "unknown",

                "cognitive_state":
                    "unknown"
            }

        dominant_emotion = (
            Counter(
                emotions
            ).most_common(1)[0][0]
        )

        dominant_stress = (
            Counter(
                stress_levels
            ).most_common(1)[0][0]
        )

        dominant_cognitive = (
            Counter(
                cognitive_states
            ).most_common(1)[0][0]
        )

        return {

            "dominant_emotion":
                dominant_emotion,

            "stress_trend":
                dominant_stress,

            "cognitive_state":
                dominant_cognitive
        }