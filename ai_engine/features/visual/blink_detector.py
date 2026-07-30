class BlinkDetector:
    def __init__(self,
                 blink_threshold=0.21,
                 reopen_threshold=0.25,
                 consecutive_frames=1):

        self.blink_threshold = blink_threshold
        self.reopen_threshold = max(reopen_threshold, blink_threshold)
        self.consecutive_frames = max(1, consecutive_frames)

        self.frame_counter = 0
        self.total_blinks = 0
        self._eye_open = True

    def detect_blink(self, ear):

        blink_detected = False

        if ear < self.blink_threshold:

            self.frame_counter += 1

            if (
                self._eye_open
                and self.frame_counter >= self.consecutive_frames
            ):
                self.total_blinks += 1
                blink_detected = True
                self._eye_open = False

        else:

            if ear >= self.reopen_threshold:
                self._eye_open = True

            self.frame_counter = 0

        return {
            "blink_detected": blink_detected,
            "total_blinks": self.total_blinks,
            "ear": ear
        }

    def reset(self):
        self.frame_counter = 0
        self.total_blinks = 0
        self._eye_open = True
