import cv2

from ai_engine.features.visual.video_frame_extractor import (
    extract_frames
)

from ai_engine.features.visual.facemesh_tracker import (
    get_face_landmarks
)

from ai_engine.features.visual.blink_detector import (
    detect_blink
)

VIDEO_PATH = (
    r"C:\Users\Sidath Mendis\Desktop\Reasearch\System-implementation"
    r"\MaCoDeR\datasets\raw\RAVDESS"
    r"\Video_Speech_Actors_01-24"
    r"\Actor_01\01-01-01-01-01-01-01.mp4"
)

frames = extract_frames(
    VIDEO_PATH,
    frame_rate=5
)

print(f"Frames Extracted: {len(frames)}")

for idx, frame in enumerate(frames):

    landmarks = get_face_landmarks(frame)

    if landmarks:

        ear = detect_blink(
            landmarks
        )

        print(
            f"Frame {idx+1} EAR:",
            round(ear, 4)
        )

    else:

        print(
            f"Frame {idx+1}: No face detected"
        )