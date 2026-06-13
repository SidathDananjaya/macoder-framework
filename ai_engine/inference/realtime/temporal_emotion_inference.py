import numpy as np
import tensorflow as tf


class TemporalEmotionInference:

    def __init__(self):

        self.model = tf.keras.models.load_model(
            "experiments/"
            "exp_006_temporal_emotion/"
            "temporal_emotion_model.h5"
        )

        self.labels = [
            "neutral",
            "calm",
            "happy",
            "sad",
            "angry",
            "fear",
            "disgust",
            "surprised"
        ]

    def predict(
        self,
        sequence
    ):

        sequence = np.expand_dims(
            sequence,
            axis=0
        )

        predictions = self.model.predict(
            sequence,
            verbose=0
        )[0]

        predicted_index = np.argmax(
            predictions
        )

        return {

            "emotion":
                self.labels[predicted_index],

            "confidence":
                float(predictions[predicted_index])
        }