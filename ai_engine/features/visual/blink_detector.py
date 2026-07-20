class BlinkDetector:
    """Eye-Aspect-Ratio (EAR) blink detector.

    Counts a blink on the *falling edge* — the instant the EAR drops below the
    close threshold from an open state — instead of waiting for the eye to
    reopen. At the realtime pipeline's ~1-2 fps a blink usually occupies at most
    a single sampled frame, so falling-edge counting registers it immediately and
    (unlike the old "count on reopen" rule) does not lose a blink that lands on
    the final frame of a session.

    Hysteresis (a separate, higher reopen threshold) stops one blink being
    counted several times when the EAR flickers around a single threshold while
    the eye is reopening, and stops a long eye-closure being counted repeatedly.

    NOTE ON ACCURACY: the counter can only see blinks that fall on a *sampled*
    frame. The vision loop is throttled by the per-frame emotion model, so at
    ~1-2 fps some sub-second blinks still happen between frames and are missed.
    Blink *rate* is therefore an approximate indicator, not a precise
    physiological count — see THESIS_RESULTS_GUIDE.md "Known measurement
    caveats". Raising the frame rate (see EMOTION_EVERY_N in
    realtime_ai_engine.py) is what actually improves capture.
    """

    def __init__(self,
                 blink_threshold=0.21,      # EAR below this  -> eye is closing
                 reopen_threshold=0.25,     # EAR above this  -> eye is open again
                 consecutive_frames=1):     # min closed frames to accept (>=1)

        self.blink_threshold = blink_threshold
        # Reopen threshold must sit at or above the close threshold for the
        # hysteresis to make sense.
        self.reopen_threshold = max(reopen_threshold, blink_threshold)
        self.consecutive_frames = max(1, consecutive_frames)

        self.frame_counter = 0      # consecutive frames currently below threshold
        self.total_blinks = 0
        self._eye_open = True       # hysteresis state: is the eye currently "open"?

    def detect_blink(self, ear):

        blink_detected = False

        if ear < self.blink_threshold:

            self.frame_counter += 1

            # Falling edge: the eye was open and has now been closed for the
            # required number of frames -> register exactly one blink and arm the
            # hysteresis so a sustained/flickering closure is not recounted.
            if (
                self._eye_open
                and self.frame_counter >= self.consecutive_frames
            ):
                self.total_blinks += 1
                blink_detected = True
                self._eye_open = False

        else:

            # Only re-arm once the eye has clearly reopened (EAR back above the
            # higher reopen threshold); values in the ambiguous band between the
            # two thresholds do not re-arm, which prevents double counts.
            if ear >= self.reopen_threshold:
                self._eye_open = True

            self.frame_counter = 0

        return {
            "blink_detected": blink_detected,
            "total_blinks": self.total_blinks,
            "ear": ear
        }

    def reset(self):
        """Clear per-session state so a new session starts counting from zero.

        The detector is created once as a module-level singleton, so without an
        explicit reset the blink total would carry over from the previous
        session within the same server process.
        """
        self.frame_counter = 0
        self.total_blinks = 0
        self._eye_open = True
