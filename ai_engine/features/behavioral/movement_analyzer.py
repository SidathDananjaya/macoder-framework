from collections import deque
import numpy as np


class MovementAnalyzer:

    def __init__(self):

        self.yaw_history = deque(maxlen=30)
        self.pitch_history = deque(maxlen=30)
        self.roll_history = deque(maxlen=30)

    def update(self, yaw, pitch, roll):

        self.yaw_history.append(yaw)
        self.pitch_history.append(pitch)
        self.roll_history.append(roll)

    def movement_intensity(self):

        if len(self.yaw_history) < 5:
            return 0

        yaw_std = np.std(self.yaw_history)
        pitch_std = np.std(self.pitch_history)
        roll_std = np.std(self.roll_history)

        movement_score = yaw_std + pitch_std + roll_std

        return round(float(movement_score), 2)