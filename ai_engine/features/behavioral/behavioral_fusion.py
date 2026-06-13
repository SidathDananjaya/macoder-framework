class BehavioralFusion:

    def build_state(
        self,
        gaze_stability,
        gaze_shifts,
        movement_score,
        cognitive_load,
        deception_risk
    ):

        return {

            "gaze_stability": gaze_stability,

            "gaze_shifts": gaze_shifts,

            "movement_score": movement_score,

            "cognitive_load": cognitive_load,

            "deception_risk": deception_risk
        }