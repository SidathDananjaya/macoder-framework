from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout
)


class TemporalEmotionModel:

    def build(
        self,
        sequence_length,
        feature_count,
        num_classes
    ):

        model = Sequential()

        model.add(
            LSTM(
                64,
                return_sequences=True,
                input_shape=(
                    sequence_length,
                    feature_count
                )
            )
        )

        model.add(
            Dropout(0.3)
        )

        model.add(
            LSTM(32)
        )

        model.add(
            Dropout(0.3)
        )

        model.add(
            Dense(
                32,
                activation="relu"
            )
        )

        model.add(
            Dense(
                num_classes,
                activation="softmax"
            )
        )

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model