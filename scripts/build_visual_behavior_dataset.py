import cv2
import pandas as pd

from pathlib import Path

from ai_engine.data.loaders.ravdess_video_loader import RavdessVideoLoader

from ai_engine.features.visual.eye_metrics import EyeMetrics
from ai_engine.features.visual.facemesh_tracker import FaceMeshTracker

from ai_engine.features.visual.head_pose_estimator import HeadPoseEstimator
from ai_engine.features.visual.gaze_tracker import GazeTracker

from ai_engine.features.behavioral.cognitive_load_estimator import CognitiveLoadEstimator
from ai_engine.features.behavioral.deception_risk_estimator import DeceptionRiskEstimator

from ai_engine.utils.ravdess_label_parser import parse_emotion_from_filename


DATASET_PATH = r"datasets/raw/RAVDESS/Video_Speech_Actors_01-24"

OUTPUT_CSV = r"datasets/processed/video/visual_behavior_features.csv"


loader = RavdessVideoLoader(DATASET_PATH)

video_files = loader.get_video_files()

facemesh = FaceMeshTracker()

eye_metrics = EyeMetrics()

headpose = HeadPoseEstimator()

gaze_tracker = GazeTracker()

cognitive_estimator = CognitiveLoadEstimator()

deception_estimator = DeceptionRiskEstimator()


all_rows = []


for video_path in video_files:

    print(f"Processing: {video_path.name}")

    emotion_label = parse_emotion_from_filename(
        video_path.name
    )

    cap = cv2.VideoCapture(str(video_path))

    ear_values = []

    yaw_values = []

    pitch_values = []

    roll_values = []

    gaze_values = []

    movement_values = []

    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        if frame_count % 10 != 0:
            continue

        results = facemesh.process_frame(frame)

        if results.multi_face_landmarks:

            face_landmarks = results.multi_face_landmarks[0]

            eye_data = eye_metrics.get_eye_metrics(
                frame,
                face_landmarks
            )

            pose = headpose.estimate_pose(
                frame,
                face_landmarks
            )

            gaze = gaze_tracker.estimate_gaze(face_landmarks)

            ear_values.append(
                eye_data["avg_ear"]
            )

            yaw_values.append(
                pose["yaw"]
            )

            pitch_values.append(
                pose["pitch"]
            )

            roll_values.append(
                pose["roll"]
            )

            movement_values.append(
                abs(pose["yaw"]) +
                abs(pose["pitch"]) +
                abs(pose["roll"])
            )

            gaze_values.append(gaze)

    cap.release()

    if len(ear_values) == 0:
        continue

    avg_ear = sum(ear_values) / len(ear_values)

    avg_yaw = sum(yaw_values) / len(yaw_values)

    avg_pitch = sum(pitch_values) / len(pitch_values)

    avg_roll = sum(roll_values) / len(roll_values)

    left_count = gaze_values.count("LEFT")
    right_count = gaze_values.count("RIGHT")
    center_count = gaze_values.count("CENTER")

    gaze_stability = center_count / len(gaze_values)

    movement_score = (
        sum(movement_values)
        / len(movement_values)
    )

    blink_rate = 15 + (1 - avg_ear) * 20

    cognitive_load = cognitive_estimator.estimate(
        blink_rate,
        emotion_label,
        gaze_stability,
        movement_score
    )

    gaze_shift_frequency = (
        left_count + right_count
    )

    deception_risk = deception_estimator.estimate(
        gaze_shift_frequency,
        movement_score,
        cognitive_load,
        emotion_label
    )

    row = {

        "emotion": emotion_label,

        "avg_ear": avg_ear,

        "avg_yaw": avg_yaw,

        "avg_pitch": avg_pitch,

        "avg_roll": avg_roll,

        "left_gaze_ratio": left_count / len(gaze_values),

        "right_gaze_ratio": right_count / len(gaze_values),

        "center_gaze_ratio": center_count / len(gaze_values),

        "cognitive_load": cognitive_load,

        "deception_risk": deception_risk,

        "blink_rate": blink_rate,

        "gaze_stability": gaze_stability,

        "movement_score": movement_score,
    }

    all_rows.append(row)

    print(row)

df = pd.DataFrame(all_rows)

Path(
    "datasets/processed/video"
).mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_CSV,
    index=False
)

print(df.head())

print(f"\nSaved to: {OUTPUT_CSV}")
