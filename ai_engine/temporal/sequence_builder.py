import numpy as np


class SequenceBuilder:

    def build_feature_vector(self, state):

        return [

            state["ear"],

            state["blink_rate"],

            state["gaze_stability"],

            state["movement_score"],

            state["yaw"],

            state["pitch"],

            state["roll"],

            state["emotion_score"],

            state["stress_score"]
        ]

    def build_sequence(self, states):

        sequence = []

        for state in states:

            vector = self.build_feature_vector(state)

            sequence.append(vector)

        return np.array(sequence, dtype=np.float32)