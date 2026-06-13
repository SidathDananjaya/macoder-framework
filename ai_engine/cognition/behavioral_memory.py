from collections import deque


class BehavioralMemory:

    def __init__(self, max_history=120):

        self.history = deque(maxlen=max_history)

    def add_state(self, state):

        self.history.append(state)

    def get_recent_states(self):

        return list(self.history)

    def average_stress(self):

        if not self.history:
            return 0.0

        values = [
            state["stress_score"]
            for state in self.history
        ]

        return sum(values) / len(values)

    def average_confidence(self):

        if not self.history:
            return 0.0

        values = [
            state["confidence"]
            for state in self.history
        ]

        return sum(values) / len(values)