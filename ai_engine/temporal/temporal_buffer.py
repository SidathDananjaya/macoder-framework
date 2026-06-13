from collections import deque


class TemporalBuffer:

    def __init__(self, max_frames=60):

        self.buffer = deque(maxlen=max_frames)

    def add_state(self, state):

        self.buffer.append(state)

    def get_sequence(self):

        return list(self.buffer)

    def is_ready(self):

        return len(self.buffer) >= self.buffer.maxlen