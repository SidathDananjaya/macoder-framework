import time
import numpy as np


class VisualStressAnalyzer:

    def __init__(self):

        self.start_time = time.time()

        self.blink_history = []
        self.ear_history = []
        self.emotion_history = []

    def update(
            self,
            ear,
            blink_count,
            emotion
    ):

        current_time = time.time()

        self.ear_history.append(ear)

        self.blink_history.append(
            (current_time, blink_count)
        )

        self.emotion_history.append(emotion)

    def calculate_blink_rate(self):

        if len(self.blink_history) < 2:
            return 0

        first_time = self.blink_history[0][0]
        last_time = self.blink_history[-1][0]

        duration = last_time - first_time

        if duration <= 0:
            return 0

        total_blinks = (
            self.blink_history[-1][1]
            -
            self.blink_history[0][1]
        )

        blink_rate = (total_blinks / duration) * 60

        return round(blink_rate, 2)

    def average_ear(self):

        if not self.ear_history:
            return 0

        return round(
            np.mean(self.ear_history),
            3
        )

    def emotion_distribution(self):

        if not self.emotion_history:
            return {}

        emotion_counts = {}

        for emotion in self.emotion_history:

            emotion_counts[emotion] = (
                    emotion_counts.get(emotion, 0)
                    + 1
            )

        total = len(self.emotion_history)

        distribution = {}

        for emotion, count in emotion_counts.items():

            distribution[emotion] = round(
                count / total,
                2
            )

        return distribution

    def estimate_stress_level(self):

        blink_rate = self.calculate_blink_rate()

        avg_ear = self.average_ear()

        stress_score = 0

        # High blink rate
        if blink_rate > 25:
            stress_score += 1

        # Eye tightening
        if avg_ear < 0.22:
            stress_score += 1

        if stress_score == 0:
            return "LOW"

        elif stress_score == 1:
            return "MEDIUM"

        else:
            return "HIGH"