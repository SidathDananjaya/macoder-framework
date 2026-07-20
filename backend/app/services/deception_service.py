from ai_engine.features.behavioral.deception_risk_estimator import (
    DeceptionRiskEstimator
)

estimator = (
    DeceptionRiskEstimator()
)


def calculate_risk(

    blink_count,
    emotion

):

    return estimator.estimate(

        gaze_shift_frequency=
            blink_count,

        movement_score=
            blink_count,

        cognitive_load=
            "LOW",

        emotion=
            emotion
    )