import numpy as np
import joblib

from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from tensorflow.keras.callbacks import EarlyStopping

from ai_engine.models.temporal.temporal_emotion_model import (
    TemporalEmotionModel
)


# ---------------------------------------------------
# Paths
# ---------------------------------------------------

X_PATH = (
    "datasets/processed/temporal/"
    "X_sequences.npy"
)

Y_PATH = (
    "datasets/processed/temporal/"
    "y_labels.npy"
)

OUTPUT_DIR = (
    "experiments/exp_006_temporal_emotion/"
)

Path(OUTPUT_DIR).mkdir(
    parents=True,
    exist_ok=True
)

# ---------------------------------------------------
# Load data
# ---------------------------------------------------

X = np.load(X_PATH)

y = np.load(Y_PATH)

print("Loaded sequences:", X.shape)

print("Loaded labels:", y.shape)

# ---------------------------------------------------
# Encode labels
# ---------------------------------------------------

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

# ---------------------------------------------------
# Split dataset
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ---------------------------------------------------
# Build model
# ---------------------------------------------------

builder = TemporalEmotionModel()

model = builder.build(
    sequence_length=60,
    feature_count=9,
    num_classes=len(encoder.classes_)
)

model.summary()

# ---------------------------------------------------
# Early stopping
# ---------------------------------------------------

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# ---------------------------------------------------
# Train
# ---------------------------------------------------

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_test,
        y_test
    ),

    epochs=30,

    batch_size=32,

    callbacks=[early_stop]
)

# ---------------------------------------------------
# Evaluate
# ---------------------------------------------------

predictions = model.predict(X_test)

predicted_classes = predictions.argmax(axis=1)

accuracy = accuracy_score(
    y_test,
    predicted_classes
)

print("\nTemporal Model Accuracy:", accuracy)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predicted_classes
    )
)

# ---------------------------------------------------
# Save model
# ---------------------------------------------------

model.save(
    f"{OUTPUT_DIR}/temporal_emotion_model.h5"
)

joblib.dump(
    encoder,
    f"{OUTPUT_DIR}/temporal_label_encoder.pkl"
)

print("\nTemporal model saved successfully")