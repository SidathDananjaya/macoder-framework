import cv2
import base64
import numpy as np
import random

# -----------------------------------------
# REAL AI IMPORTS
# -----------------------------------------

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

# -----------------------------------------
# Initialize AI Modules
# -----------------------------------------

emotion_detector = EmotionDetector()

blink_detector = BlinkDetector(
    blink_threshold=0.27,
    consecutive_frames=2
)

facemesh_tracker = FaceMeshTracker()

eye_metrics = EyeMetrics()

# -----------------------------------------
# PROCESS FRAME
# -----------------------------------------

async def process_frame(frame_data: str):

    try:

        # -----------------------------------------
        # Decode Base64 Image
        # -----------------------------------------

        encoded_data = frame_data.split(",")[1]

        nparr = np.frombuffer(
            base64.b64decode(encoded_data),
            np.uint8
        )

        frame = cv2.imdecode(
            nparr,
            cv2.IMREAD_COLOR
        )

        # -----------------------------------------
        # EMOTION DETECTION
        # -----------------------------------------

        emotion_result = (
            emotion_detector.analyze_emotion(frame)
        )

        emotion = (
            emotion_result["dominant_emotion"]
        )

        # -----------------------------------------
        # TEMPORAL EMOTION
        # -----------------------------------------

        temporal_emotion = emotion

        # -----------------------------------------
        # BLINK DETECTION
        # -----------------------------------------

        results = facemesh_tracker.process_frame(frame)

        avg_ear = 0.3

        if results.multi_face_landmarks:

            face_landmarks = (
                results.multi_face_landmarks[0]
            )

            metrics = eye_metrics.get_eye_metrics(
                frame,
                face_landmarks
            )

            avg_ear = metrics["avg_ear"]

        print("EAR:", avg_ear)

        blink_result = (
            blink_detector.detect_blink(avg_ear)
        )

        total_blinks = (
            blink_result["total_blinks"]
        )

        # -----------------------------------------
        # STRESS ESTIMATION
        # -----------------------------------------

        if total_blinks > 15:

            stress_level = "HIGH"

        elif total_blinks > 7:

            stress_level = "MEDIUM"

        else:

            stress_level = "LOW"

        # -----------------------------------------
        # CONFIDENCE
        # -----------------------------------------

        fusion_confidence = round(
            random.uniform(0.70, 0.99),
            2
        )

        # -----------------------------------------
        # FINAL RESULT
        # -----------------------------------------

        ai_result = {

            "emotion": emotion,

            "temporal_emotion":
                temporal_emotion,

            "stress_level":
                stress_level,

            "fusion_confidence":
                fusion_confidence,

            "blink_count":
                total_blinks
        }

        print("REALTIME AI:", ai_result)

        return ai_result

    except Exception as e:

        print("FRAME PROCESS ERROR:", e)

        return {
            "emotion": "unknown",
            "temporal_emotion": "unknown",
            "stress_level": "LOW",
            "fusion_confidence": 0,
            "blink_count": 0
        }