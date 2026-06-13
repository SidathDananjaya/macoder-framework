EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


def parse_emotion_from_filename(filename):

    parts = filename.split("-")

    emotion_code = parts[2]

    return EMOTION_MAP.get(emotion_code, "unknown")