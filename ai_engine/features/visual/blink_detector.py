class BlinkDetector:

    def __init__(self,
                 blink_threshold=0.20,
                 consecutive_frames=4):

        self.blink_threshold = blink_threshold
        self.consecutive_frames = consecutive_frames

        self.frame_counter = 0
        self.total_blinks = 0

    def detect_blink(self, ear):

        blink_detected = False

        if ear < self.blink_threshold:

            self.frame_counter += 1

        else:

            if self.frame_counter >= self.consecutive_frames:

                self.total_blinks += 1
                blink_detected = True

            self.frame_counter = 0

        return {
            "blink_detected": blink_detected,
            "total_blinks": self.total_blinks,
            "ear": ear
        }