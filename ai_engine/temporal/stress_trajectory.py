from collections import deque
import numpy as np


class StressTrajectory:

    def __init__(self):

        self.history = deque(maxlen=120)

    def update(self, stress_score):

        self.history.append(stress_score)

    def trend(self):

        if len(self.history) < 10:
            return "STABLE"

        first_half = list(self.history)[:len(self.history)//2]
        second_half = list(self.history)[len(self.history)//2:]

        first_avg = np.mean(first_half)
        second_avg = np.mean(second_half)

        if second_avg > first_avg + 0.5:
            return "INCREASING"

        elif second_avg < first_avg - 0.5:
            return "DECREASING"

        return "STABLE"