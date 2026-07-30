import glob
import os
import sys

import cv2
import numpy as np
import pandas as pd
import soundfile as sf

from ai_engine.features.visual.facemesh_tracker import FaceMeshTracker
from ai_engine.features.visual.eye_metrics import EyeMetrics
from ai_engine.features.visual.head_pose_estimator import HeadPoseEstimator
from ai_engine.features.visual.gaze_tracker import GazeTracker
from ai_engine.features.audio.timing_features import extract_timing_features
from ai_engine.utils.ravdess_label_parser import parse_emotion_from_filename

VIDEO_ROOT = "datasets/raw/RAVDESS/Video_Speech_Actors_01-24"
AUDIO_ROOT = "datasets/raw/RAVDESS/Audio_Speech_Actors_01-24"
AUDIO_CSV = "datasets/processed/audio/stress_audio_features.csv"
OUT_CSV = "datasets/processed/multimodal/aligned_features.csv"

FRAME_STEP = 10

facemesh = FaceMeshTracker()
eye_metrics = EyeMetrics()
headpose = HeadPoseEstimator()
gaze_tracker = GazeTracker()


def _code_key(filename):
    parts = os.path.basename(filename).replace(".wav", "").replace(".mp4", "").split("-")
    return "-".join(parts[2:7]) if len(parts) >= 7 else None


def _audio_feature_lookup():
    df = pd.read_csv(AUDIO_CSV)
    drop = [c for c in ("file", "emotion", "stress_level") if c in df.columns]
    feats = df.drop(columns=drop).select_dtypes(include=[np.number])
    lookup = {}
    for i, path in enumerate(df["file"]):
        key = _code_key(path)
        if key:
            lookup[key] = {f"aud_{c}": feats.iloc[i][c] for c in feats.columns}
    return lookup


def _visual_features(video_path):
    cap = cv2.VideoCapture(str(video_path))
    ears, yaws, pitches, rolls, moves, gazes = [], [], [], [], [], []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % FRAME_STEP != 0:
            continue
        res = facemesh.process_frame(frame)
        if not res.multi_face_landmarks:
            continue
        lm = res.multi_face_landmarks[0]
        pose = headpose.estimate_pose(frame, lm)
        ears.append(eye_metrics.get_eye_metrics(frame, lm)["avg_ear"])
        yaws.append(pose["yaw"]); pitches.append(pose["pitch"]); rolls.append(pose["roll"])
        moves.append(abs(pose["yaw"]) + abs(pose["pitch"]) + abs(pose["roll"]))
        gazes.append(gaze_tracker.estimate_gaze(lm))
    cap.release()

    if not ears:
        return None

    n = len(gazes)
    avg_ear = float(np.mean(ears))
    return {
        "vis_avg_ear": avg_ear,
        "vis_avg_yaw": float(np.mean(yaws)),
        "vis_avg_pitch": float(np.mean(pitches)),
        "vis_avg_roll": float(np.mean(rolls)),
        "vis_left_gaze_ratio": gazes.count("LEFT") / n,
        "vis_right_gaze_ratio": gazes.count("RIGHT") / n,
        "vis_center_gaze_ratio": gazes.count("CENTER") / n,
        "vis_gaze_stability": gazes.count("CENTER") / n,
        "vis_movement_score": float(np.mean(moves)),
        "vis_blink_rate": 15 + (1 - avg_ear) * 20,
    }


def _timing_features(code, actor_dir):
    matches = glob.glob(os.path.join(AUDIO_ROOT, actor_dir, f"03-01-{code}.wav"))
    if not matches:
        return {f"tim_{k}": 0.0 for k in
                ("response_latency", "pause_ratio", "speech_rate", "voiced_ratio", "mean_pause")}
    sig, sr = sf.read(matches[0])
    if sig.ndim > 1:
        sig = sig.mean(axis=1)
    t = extract_timing_features(sig.astype(np.float32), sr)
    return {f"tim_{k}": v for k, v in t.items()}


def main():
    n_actors = int(sys.argv[1]) if len(sys.argv) > 1 else 8

    print("Loading acoustic feature lookup ...")
    aud_lookup = _audio_feature_lookup()

    actor_dirs = sorted(os.listdir(VIDEO_ROOT))[:n_actors]
    print(f"Building aligned dataset from {len(actor_dirs)} actors ...")

    rows = []
    for adir in actor_dirs:
        videos = sorted(glob.glob(os.path.join(VIDEO_ROOT, adir, "*.mp4")))
        for vp in videos:
            code = _code_key(vp)
            aud = aud_lookup.get(code)
            if aud is None:
                continue
            vis = _visual_features(vp)
            if vis is None:
                continue
            tim = _timing_features(code, adir)
            row = {"emotion": parse_emotion_from_filename(os.path.basename(vp)),
                   "actor": adir}
            row.update(aud); row.update(vis); row.update(tim)
            rows.append(row)
        print(f"  {adir}: {len(rows)} rows so far")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} aligned utterances -> {OUT_CSV}")
    print("Columns:", sum(c.startswith('aud_') for c in df.columns), "audio,",
          sum(c.startswith('vis_') for c in df.columns), "visual,",
          sum(c.startswith('tim_') for c in df.columns), "timing")


if __name__ == "__main__":
    main()
