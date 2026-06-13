STRESS_MAP = {
    "angry": "high",
    "fearful": "high",

    "disgust": "medium",
    "sad": "medium",
    "surprised": "medium",

    "calm": "low",
    "neutral": "low",
    "happy": "low"
}


def map_emotion_to_stress(emotion):

    return STRESS_MAP.get(emotion, "unknown")