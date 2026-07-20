"""Live temporal-emotion inference (dissertation Section 5.4.5).

Wraps the honest, retrained LSTM (`temporal_emotion_model_v2.h5`, 7 GENUINE
behavioural features - no leaked `emotion_score`) so the deployed pipeline emits
a real sequence-based ``temporal_emotion`` instead of just echoing the current
frame's emotion. (The previous version loaded the leaky 9-feature model with a
hard-coded, mis-ordered label list and was never wired into the live loop.)

It keeps a rolling buffer of the last ``SEQUENCE_LENGTH`` feature vectors; once
full it predicts on the sequence, otherwise it falls back to the current frame
emotion. The per-frame feature vector matches how the training sequences were
built (build_temporal_sequences.py), including the derived ``blink_rate`` proxy.
"""

import os
from collections import deque

import numpy as np

SEQUENCE_LENGTH = 60
MODEL_PATH = "experiments/exp_006_temporal_emotion/temporal_emotion_model_v2.h5"
ENCODER_PATH = "experiments/exp_006_temporal_emotion/temporal_label_encoder_v2.pkl"


class TemporalEmotionInference:
    """Rolling-window LSTM emotion smoother/predictor."""

    def __init__(self):
        self._buffer = deque(maxlen=SEQUENCE_LENGTH)
        self._model = None
        self._encoder = None
        self._load()

    def _load(self):
        try:
            import joblib
            from tensorflow.keras.models import load_model

            if os.path.exists(MODEL_PATH):
                self._model = load_model(MODEL_PATH, compile=False)
                self._encoder = joblib.load(ENCODER_PATH)
            else:
                print("TEMPORAL MODEL not found (run retrain_temporal); "
                      "falling back to frame emotion.")
        except Exception as error:  # noqa: BLE001 - never break the pipeline
            print("TEMPORAL MODEL LOAD ERROR:", error)
            self._model = None

    def reset(self):
        self._buffer.clear()

    def predict(self, avg_ear, yaw, pitch, roll, gaze_stability,
                movement_score, frame_emotion):
        """Append the current frame's features and return a temporal emotion.

        Falls back to ``frame_emotion`` until the buffer is full or if the model
        is unavailable.
        """

        # Same 7 genuine features (and derived blink_rate) as the training set.
        blink_rate = 15.0 + (1.0 - avg_ear) * 20.0
        self._buffer.append([
            avg_ear, yaw, pitch, roll,
            blink_rate, gaze_stability, movement_score,
        ])

        if self._model is None or len(self._buffer) < SEQUENCE_LENGTH:
            return frame_emotion

        try:
            seq = np.asarray(self._buffer, dtype=np.float32)[np.newaxis, ...]
            probs = self._model.predict(seq, verbose=0)[0]
            return str(self._encoder.inverse_transform([int(np.argmax(probs))])[0])
        except Exception as error:  # noqa: BLE001
            print("TEMPORAL INFERENCE ERROR:", error)
            return frame_emotion
