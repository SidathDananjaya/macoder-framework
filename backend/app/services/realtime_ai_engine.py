import cv2
import base64
import numpy as np
import random

from collections import deque

# AI feature/inference imports.

from ai_engine.features.quality.quality_assessor import (
    assess_video_quality
)

from ai_engine.features.visual.emotion_detector import (
    EmotionDetector
)

from ai_engine.features.visual.blink_detector import (
    BlinkDetector
)

from ai_engine.features.visual.facemesh_tracker import (
    FaceMeshTracker
)

from ai_engine.features.visual.eye_metrics import (
    EyeMetrics
)

from ai_engine.features.visual.head_pose_estimator import (
    HeadPoseEstimator
)

from ai_engine.features.visual.gaze_tracker import (

GazeTracker

)

from ai_engine.features.behavioral.gaze_behavior import (

GazeBehaviorAnalyzer

)

from ai_engine.features.behavioral.movement_analyzer import (
    MovementAnalyzer
)

from ai_engine.inference.realtime.cognitive_state_inference import (
    CognitiveStateInference
)

from ai_engine.inference.realtime.temporal_emotion_inference import (
    TemporalEmotionInference
)

# Initialise AI modules.

emotion_detector = EmotionDetector()

# The loop runs at ~1 fps but a blink lasts only 0.1-0.4 s, so count on a single frame with the standard EAR threshold (~0.21); the old 0.27/2 config needed a ~1 s closure and missed most blinks.
blink_detector = BlinkDetector(
    blink_threshold=0.21,
    consecutive_frames=1
)

facemesh_tracker = FaceMeshTracker()

eye_metrics = EyeMetrics()

head_pose_estimator = HeadPoseEstimator()

movement_analyzer = MovementAnalyzer()

gaze_tracker = GazeTracker()

gaze_behavior = GazeBehaviorAnalyzer()

cognitive_inference = (
    CognitiveStateInference()
)

# Sequence-based temporal emotion (7-feature LSTM); falls back to the frame emotion until its 60-frame buffer fills.
temporal_inference = (
    TemporalEmotionInference()
)

from ai_engine.features.behavioral.cognitive_load_estimator import (
    CognitiveLoadEstimator
)

from ai_engine.features.behavioral.deception_risk_estimator import (
    DeceptionRiskEstimator
)

cognitive_load_estimator = (
    CognitiveLoadEstimator()
)

deception_estimator = (
    DeceptionRiskEstimator()
)

# DeepFace is the ~0.5-0.8 s bottleneck, so refresh emotion every EMOTION_EVERY_N frames and reuse the cached label; the cheap blink path still runs every frame and catches more blinks. Set to 1 for per-frame emotion.
EMOTION_EVERY_N = 2

_frame_index = 0
_cached_emotion = {"dominant_emotion": "neutral", "emotion_scores": {}}


# Recent-window multi-signal stress (replaces the old cumulative blink count that could only rise): stress = 0.5*recent-blink-rate + 0.3*head-movement + 0.2*emotion-negativity, over ~20 frames (~13 s).
STRESS_WINDOW_FRAMES = 20
STRESS_BLINK_CEILING = 8.0      # this many blinks in the window reads as maximal
STRESS_HIGH_THRESHOLD = 0.60
STRESS_MEDIUM_THRESHOLD = 0.33
_NEGATIVE_EMOTIONS = {"fear", "angry", "sad", "disgust"}

_blink_window = deque(maxlen=STRESS_WINDOW_FRAMES)


def _estimate_visual_stress(blink_detected, movement_score, emotion):
    """Recent-window multi-signal stress level (LOW / MEDIUM / HIGH)."""

    _blink_window.append(1 if blink_detected else 0)

    # Blink rate over the recent window (not a running total) so stress falls again when blinking calms.
    blink_component = min(1.0, sum(_blink_window) / STRESS_BLINK_CEILING)

    movement_component = min(1.0, (movement_score or 0.0) / 20.0)

    if emotion in _NEGATIVE_EMOTIONS:
        emotion_component = 1.0
    elif emotion == "neutral":
        emotion_component = 0.2
    else:
        emotion_component = 0.0

    score = (
        0.5 * blink_component
        + 0.3 * movement_component
        + 0.2 * emotion_component
    )

    if score >= STRESS_HIGH_THRESHOLD:
        return "HIGH"
    if score >= STRESS_MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def _face_region_quality(frame, face_landmarks):
    # Measure video quality on the face crop, not the whole frame, because the background would otherwise dominate the score (e.g. a small webcam window on a big OBS canvas).

    height, width = frame.shape[:2]

    xs = [lm.x for lm in face_landmarks.landmark]
    ys = [lm.y for lm in face_landmarks.landmark]

    x1 = int(max(0.0, min(xs)) * width)
    x2 = int(min(1.0, max(xs)) * width)
    y1 = int(max(0.0, min(ys)) * height)
    y2 = int(min(1.0, max(ys)) * height)

    # Pad the box ~15% to include the whole face, not just the landmarks.
    pad_x = int(0.15 * max(1, x2 - x1))
    pad_y = int(0.15 * max(1, y2 - y1))

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)

    face_crop = frame[y1:y2, x1:x2]

    if face_crop.size == 0:
        return assess_video_quality(frame)

    return assess_video_quality(face_crop)


def reset_engine():
    # Reset per-session state (blink total, emotion cache, movement/gaze) so each new session starts clean; these are module-level singletons that would otherwise carry over.

    global _frame_index, _cached_emotion

    blink_detector.reset()

    temporal_inference.reset()

    _blink_window.clear()

    _frame_index = 0
    _cached_emotion = {"dominant_emotion": "neutral", "emotion_scores": {}}

    # Rolling movement/gaze analysers also accumulate across frames, so reset them too when supported.
    for _mod in (movement_analyzer, gaze_behavior):
        if hasattr(_mod, "reset"):
            _mod.reset()


# Process one incoming frame.

async def process_frame(frame_data: str):

    try:

        # Decode the base64 image.
        encoded_data = frame_data.split(",")[1]

        nparr = np.frombuffer(
            base64.b64decode(encoded_data),
            np.uint8
        )

        frame = cv2.imdecode(
            nparr,
            cv2.IMREAD_COLOR
        )

        # Signal quality (Section 5.4.2) drives the fusion weights and warning; this is a whole-frame fallback, replaced by a face-region measurement once the face is found below.

        video_quality = assess_video_quality(frame)

        # Emotion detection (throttled): run the heavy model every Nth frame and reuse the cached label, but always run the first frame so we never serve an empty default.

        global _frame_index, _cached_emotion

        _frame_index += 1

        if (
            (_frame_index - 1) % EMOTION_EVERY_N == 0
            or not _cached_emotion.get("emotion_scores")
        ):
            _cached_emotion = (
                emotion_detector.analyze_emotion(frame)
            )

        emotion_result = _cached_emotion

        emotion = (
            emotion_result["dominant_emotion"]
        )

        # Visual emotion confidence from the detector (DeepFace scores are 0..100 percentages).
        emotion_scores = (
            emotion_result.get("emotion_scores", {})
        )

        visual_emotion_confidence = (
            emotion_scores.get(emotion, 0.0) / 100.0
            if emotion_scores else 0.0
        )

        # Temporal emotion is computed after the visual features below, since the LSTM needs EAR/pose/gaze.

        # Blink detection.
        results = facemesh_tracker.process_frame(frame)

        avg_ear = 0.3

        movement_score = 0

        gaze_direction = "CENTER"

        gaze_stability = 1.0

        gaze_shift_frequency = 0

        yaw = 0

        pitch = 0

        roll = 0

        if results.multi_face_landmarks:

            face_landmarks = (
                results.multi_face_landmarks[0]
            )

            # Video quality on the face region.
            video_quality = _face_region_quality(
                frame,
                face_landmarks
            )

            # Eye metrics.
            metrics = eye_metrics.get_eye_metrics(
                frame,
                face_landmarks
            )

            avg_ear = metrics["avg_ear"]

            # Head pose.
            pose = (
                head_pose_estimator
                .estimate_pose(
                    frame,
                    face_landmarks
                )
            )

            yaw = pose["yaw"]
            pitch = pose["pitch"]
            roll = pose["roll"]

            # Head movement.
            movement_analyzer.update(
                yaw,
                pitch,
                roll
            )

            movement_score = (
                movement_analyzer
                .movement_intensity()
            )

            # Gaze tracking.
            gaze_direction = (

                gaze_tracker.estimate_gaze(

                    face_landmarks

                )

            )

            gaze_behavior.update(gaze_direction)

            gaze_stability = (

                gaze_behavior.gaze_stability_score()

            )

            gaze_shift_frequency = (

                gaze_behavior.gaze_shift_frequency()

            )

        print("EAR:", avg_ear)

        blink_result = (
            blink_detector.detect_blink(avg_ear)
        )

        total_blinks = (
            blink_result["total_blinks"]
        )

        # Temporal emotion from the sequence LSTM over the last 60 frames.
        temporal_emotion = temporal_inference.predict(
            avg_ear=avg_ear,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            gaze_stability=gaze_stability,
            movement_score=movement_score,
            frame_emotion=emotion,
        )

        # Stress from recent blink rate + head movement + emotion negativity over a rolling window, so it can rise and fall.
        stress_level = _estimate_visual_stress(
            blink_detected=blink_result.get("blink_detected", False),
            movement_score=movement_score,
            emotion=emotion,
        )

        # Cognitive load.
        cognitive_load = (
            cognitive_load_estimator.estimate(
                blink_rate=total_blinks,
                emotion=emotion,
                gaze_stability=gaze_stability,
                movement_score=movement_score
            )
        )

        # Deception risk.
        deception_risk = (
            deception_estimator.estimate(
                gaze_shift_frequency=
                gaze_shift_frequency,

                movement_score=
                movement_score,

                cognitive_load=
                cognitive_load,

                emotion=
                emotion
            )
        )

        # Cognitive-state model input.
        stress_score = {

            "LOW": 1,
            "MEDIUM": 2,
            "HIGH": 3

        }[stress_level]

        features = {

            "avg_ear": avg_ear,

            "blink_rate": total_blinks,

            "gaze_stability": gaze_stability,

            "movement_score": movement_score,

            "stress_score": stress_score,

            "cognitive_load":
                {
                    "LOW": 1,
                    "MEDIUM": 2,
                    "HIGH": 3
                }[cognitive_load],

            "deception_risk":
                {
                    "LOW": 1,
                    "MEDIUM": 2,
                    "HIGH": 3
                }[deception_risk]
        }

        cognitive_result = (
            cognitive_inference.predict(features)
        )

        cognitive_state = (
            cognitive_result["state"]
        )

        cognitive_confidence = (
            cognitive_result["confidence"]
        )

        # Visual-only emotion here; the authoritative multimodal fusion runs later once audio is available.
        final_emotion = emotion

        fusion_confidence = visual_emotion_confidence

        temporal_state = "STABLE"

        # Final result sent back to the client.
        ai_result = {

            "emotion":
                emotion,

            "visual_emotion_confidence":
                round(visual_emotion_confidence, 2),

            "temporal_emotion":
                temporal_emotion,

            "final_emotion":
                final_emotion,

            "stress_level":
                stress_level,

            "visual_stress":
                stress_level,

            "cognitive_load":
                cognitive_load,

            "deception_risk":
                deception_risk,

            "visual_cognitive_state":
                cognitive_state,

            "cognitive_confidence":
                round(
                    cognitive_confidence,
                    2
                ),

            "fusion_confidence":
                round(
                    fusion_confidence,
                    2
                ),

            "temporal_state":
                temporal_state,

            "blink_count":
                total_blinks,

            "movement_score":
                round(
                    movement_score,
                    2
                ),

            "gaze_direction": gaze_direction,

            "gaze_stability": gaze_stability,

            "gaze_shift_frequency": gaze_shift_frequency,

            "video_quality":
                round(video_quality, 3),

            "head_pose": {

                "yaw":
                    round(yaw, 2),

                "pitch":
                    round(pitch, 2),

                "roll":
                    round(roll, 2)
            }
        }

        print("REALTIME AI:", ai_result)

        print()

        print("HEAD POSE")

        print("Yaw :", round(yaw,2))

        print("Pitch :", round(pitch,2))

        print("Roll :", round(roll,2))

        print("Movement :", movement_score)

        print()

        return ai_result

    except Exception as e:

        print("FRAME PROCESS ERROR:", e)

        return {

            "emotion": "unknown",

            "temporal_emotion": "unknown",

            "stress_level": "LOW",

            "cognitive_state": "UNKNOWN",

            "cognitive_confidence": 0,

            "fusion_confidence": 0,

            "blink_count": 0,

            "video_quality": 0.0
        }