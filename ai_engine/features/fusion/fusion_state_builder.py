class FusionStateBuilder:

    def build(

        self,

        ear,
        blink_rate,
        yaw,
        pitch,
        roll,
        movement_score,
        gaze_stability,
        cognitive_load_score,
        deception_risk_score
    ):

        return {

            "ear": ear,

            "blink_rate": blink_rate,

            "yaw": yaw,

            "pitch": pitch,

            "roll": roll,

            "movement_score": movement_score,

            "gaze_stability": gaze_stability,

            "cognitive_load_score": cognitive_load_score,

            "deception_risk_score": deception_risk_score
        }