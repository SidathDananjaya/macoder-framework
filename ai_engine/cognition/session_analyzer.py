class SessionAnalyzer:

    def summarize(

        self,

        average_stress,
        anomaly_count,
        dominant_emotion

    ):

        return {

            "average_stress": average_stress,

            "anomaly_events": anomaly_count,

            "dominant_emotion": dominant_emotion
        }