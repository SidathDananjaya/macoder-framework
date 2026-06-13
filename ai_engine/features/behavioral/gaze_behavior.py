from collections import deque


class GazeBehaviorAnalyzer:

    def __init__(self):

        self.gaze_history = deque(maxlen=60)

    def update(self, gaze_direction):

        self.gaze_history.append(gaze_direction)

    def gaze_stability_score(self):

        if len(self.gaze_history) < 10:
            return 1.0

        center_count = self.gaze_history.count("CENTER")

        stability = center_count / len(self.gaze_history)

        return round(stability, 2)

    def gaze_shift_frequency(self):

        if len(self.gaze_history) < 2:
            return 0

        shifts = 0

        previous = self.gaze_history[0]

        for current in list(self.gaze_history)[1:]:

            if current != previous:
                shifts += 1

            previous = current

        return shifts