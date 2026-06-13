import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def build_lstm_model(input_shape):

    model = Sequential()

    model.add(
        LSTM(
            64,
            return_sequences=True,
            input_shape=input_shape
        )
    )

    model.add(Dropout(0.3))

    model.add(LSTM(32))

    model.add(Dropout(0.3))

    model.add(Dense(16, activation='relu'))

    model.add(Dense(3, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model