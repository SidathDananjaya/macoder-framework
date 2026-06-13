import cv2

from ai_engine.features.visual.face_detector import FaceDetector
from ai_engine.features.visual.facemesh_tracker import FaceMeshTracker

from ai_engine.features.visual.eye_metrics import EyeMetrics
from ai_engine.features.visual.blink_detector import BlinkDetector

from ai_engine.features.visual.emotion_detector import EmotionDetector

from ai_engine.features.visual.visual_stress_analyzer import VisualStressAnalyzer

from ai_engine.features.visual.head_pose_estimator import HeadPoseEstimator
from ai_engine.features.visual.gaze_tracker import GazeTracker

from ai_engine.features.behavioral.gaze_behavior import GazeBehaviorAnalyzer
from ai_engine.features.behavioral.movement_analyzer import MovementAnalyzer
from ai_engine.features.behavioral.cognitive_load_estimator import CognitiveLoadEstimator
from ai_engine.features.behavioral.deception_risk_estimator import DeceptionRiskEstimator
from ai_engine.features.behavioral.behavioral_fusion import BehavioralFusion

from ai_engine.temporal.temporal_buffer import TemporalBuffer
from ai_engine.temporal.sequence_builder import SequenceBuilder
from ai_engine.temporal.stress_trajectory import StressTrajectory

from ai_engine.inference.realtime.visual_behavior_inference import VisualBehaviorInference

from ai_engine.fusion.multimodal_fusion_engine import (
    MultimodalFusionEngine
)

from ai_engine.inference.realtime.temporal_emotion_inference import (
    TemporalEmotionInference
)

from ai_engine.cognition.behavioral_memory import (
    BehavioralMemory
)

from ai_engine.cognition.stress_pattern_analyzer import (
    StressPatternAnalyzer
)

from ai_engine.cognition.anomaly_detector import (
    AnomalyDetector
)

from ai_engine.cognition.confidence_fusion import (
    ConfidenceFusion
)

from ai_engine.cognition.cognitive_state_engine import (
    CognitiveStateEngine
)

from ai_engine.cognition.session_analyzer import (
    SessionAnalyzer
)

from ai_engine.analytics.session_logger import (
    SessionLogger
)

from ai_engine.analytics.risk_scoring_engine import (
    RiskScoringEngine
)

from ai_engine.analytics.session_report_generator import (
    SessionReportGenerator
)

class VideoProcessor:

    def __init__(self, camera_index=0):

        self.camera_index = camera_index
        self.cap = None

        self.face_detector = FaceDetector()
        self.facemesh = FaceMeshTracker()

        self.eye_metrics = EyeMetrics()
        self.blink_detector = BlinkDetector()

        self.emotion_detector = EmotionDetector()

        # Performance optimization
        self.frame_count = 0
        self.current_emotion = "unknown"

        self.visual_analyzer = VisualStressAnalyzer()

        self.blink_rate = 0
        self.stress_level = "LOW"

        self.head_pose = HeadPoseEstimator()
        self.gaze_tracker = GazeTracker()

        self.gaze_behavior = GazeBehaviorAnalyzer()

        self.movement_analyzer = MovementAnalyzer()

        self.cognitive_estimator = CognitiveLoadEstimator()

        self.deception_estimator = DeceptionRiskEstimator()

        self.behavioral_fusion = BehavioralFusion()

        self.temporal_buffer = TemporalBuffer()

        self.sequence_builder = SequenceBuilder()

        self.stress_trajectory = StressTrajectory()

        self.visual_inference = VisualBehaviorInference()

        self.predicted_emotion = "unknown"

        self.prediction_confidence = 0.0

        self.temporal_inference = (
            TemporalEmotionInference()
        )

        self.multimodal_fusion = (
            MultimodalFusionEngine()
        )

        self.temporal_prediction = "unknown"

        self.temporal_confidence = 0.0

        self.final_state = ""

        self.behavioral_memory = BehavioralMemory()

        self.stress_pattern_analyzer = (
            StressPatternAnalyzer()
        )

        self.anomaly_detector = (
            AnomalyDetector()
        )

        self.confidence_fusion = (
            ConfidenceFusion()
        )

        self.cognitive_engine = (
            CognitiveStateEngine()
        )

        self.session_analyzer = (
            SessionAnalyzer()
        )

        self.final_confidence = 0.0

        self.anomaly_risk = "LOW"

        self.cognitive_assessment = ""

        self.session_logger = SessionLogger()

        self.risk_engine = RiskScoringEngine()

        self.report_generator = (
            SessionReportGenerator()
        )

        self.risk_score = 0

        self.risk_level = "NORMAL"
        

    def start_camera(self):

        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise Exception("Cannot access webcam")

        print("Webcam started successfully")

    def process_stream(self):

        while True:

            ret, frame = self.cap.read()

            if not ret:
                print("Failed to read frame")
                break

            frame = cv2.resize(frame, (960, 540))

            self.frame_count += 1

            # Detect faces
            detections = self.face_detector.detect_faces(frame)

            # Draw detections
            frame = self.face_detector.draw_detections(
                frame,
                detections
            )

            results = self.facemesh.process_frame(frame)

            if results.multi_face_landmarks:

                for face_landmarks in results.multi_face_landmarks:

                    eye_data = self.eye_metrics.get_eye_metrics(
                        frame,
                        face_landmarks
                    )

                    eye_width = eye_data["eye_width"]

                    if eye_width > 80:

                        blink_data = self.blink_detector.detect_blink(
                            eye_data["avg_ear"]
                        )

                    else:

                        blink_data = {
                            "total_blinks": self.blink_detector.total_blinks
                        }

                    ear = eye_data["avg_ear"]
                    total_blinks = blink_data["total_blinks"]

                    self.visual_analyzer.update(
                        ear,
                        total_blinks,
                        self.current_emotion
                    )

                    pose = self.head_pose.estimate_pose(
                        frame,
                        face_landmarks
                    )

                    gaze = self.gaze_tracker.estimate_gaze(
                        face_landmarks
                    )

                    pitch = pose["pitch"]
                    yaw = pose["yaw"]
                    roll = pose["roll"]

                    gaze_direction = gaze

                    self.gaze_behavior.update(gaze_direction)

                    self.movement_analyzer.update(
                        yaw,
                        pitch,
                        roll
                    )

                    gaze_stability = (
                        self.gaze_behavior.gaze_stability_score()
                    )

                    gaze_shifts = (
                        self.gaze_behavior.gaze_shift_frequency()
                    )

                    movement_score = (
                        self.movement_analyzer.movement_intensity()
                    )

                    cognitive_load = self.cognitive_estimator.estimate(
                        self.blink_rate,
                        self.current_emotion,
                        gaze_stability,
                        movement_score
                    )

                    deception_risk = self.deception_estimator.estimate(
                        gaze_shifts,
                        movement_score,
                        cognitive_load,
                        self.current_emotion
                    )

                    behavior_state = self.behavioral_fusion.build_state(
                        gaze_stability,
                        gaze_shifts,
                        movement_score,
                        cognitive_load,
                        deception_risk
                    )

                    emotion_score = 0.5
                    stress_score = 0.5

                    state = {

                        "ear": ear,

                        "blink_rate": self.blink_rate,

                        "gaze_stability": gaze_stability,

                        "movement_score": movement_score,

                        "yaw": yaw,

                        "pitch": pitch,

                        "roll": roll,

                        "emotion_score": emotion_score,

                        "stress_score": stress_score
                    }

                    self.temporal_buffer.add_state(state)

                    self.stress_trajectory.update(stress_score)

                    stress_trend = self.stress_trajectory.trend()

                    if self.temporal_buffer.is_ready():

                        states = self.temporal_buffer.get_sequence()

                        sequence = self.sequence_builder.build_sequence(
                            states
                        )

                        temporal_prediction = (
                            self.temporal_inference.predict(
                                sequence
                            )
                        )

                        self.temporal_prediction = (
                            temporal_prediction["emotion"]
                        )

                        self.temporal_confidence = (
                            temporal_prediction["confidence"] * 100
                        )

                        fusion_result = (
                            self.multimodal_fusion.process(
                                self.predicted_emotion,
                                self.prediction_confidence / 100,

                                self.temporal_prediction,
                                self.temporal_confidence / 100,

                                self.stress_level,
                                cognitive_load,
                                deception_risk
                            )
                        )

                        self.final_state = (
                            fusion_result["cognitive_state"]
                        )

                        stress_numeric = {

                            "LOW": 0.2,
                            "MEDIUM": 0.5,
                            "HIGH": 0.9

                        }.get(self.stress_level, 0.5)

                        memory_state = {

                            "stress_score": stress_numeric,

                            "confidence":
                                self.temporal_confidence / 100
                        }

                        self.behavioral_memory.add_state(
                            memory_state
                        )

                        anomaly = self.anomaly_detector.detect(

                            gaze_stability,

                            movement_score,

                            self.blink_rate,

                            self.stress_level
                        )

                        self.anomaly_risk = anomaly["risk"]

                        self.final_confidence = (
                            self.confidence_fusion.calculate(

                                self.prediction_confidence / 100,

                                self.temporal_confidence / 100,

                                self.anomaly_risk
                            )
                        )

                        self.cognitive_assessment = (
                            self.cognitive_engine.evaluate(

                                self.stress_level,

                                cognitive_load,

                                deception_risk,

                                self.anomaly_risk,

                                self.temporal_prediction
                            )
                        )

                        risk_result = self.risk_engine.calculate(

                            self.stress_level,

                            cognitive_load,

                            deception_risk,

                            self.anomaly_risk
                        )

                        self.risk_score = (
                            risk_result["score"]
                        )

                        self.risk_level = (
                            risk_result["level"]
                        )

                        self.session_logger.log({

                            "emotion":
                                self.predicted_emotion,

                            "emotion_confidence":
                                self.prediction_confidence,

                            "temporal_emotion":
                                self.temporal_prediction,

                            "temporal_confidence":
                                self.temporal_confidence,

                            "stress_level":
                                self.stress_level,

                            "cognitive_load":
                                cognitive_load,

                            "deception_risk":
                                deception_risk,

                            "gaze":
                                gaze,

                            "yaw":
                                yaw,

                            "pitch":
                                pitch,

                            "roll":
                                roll,

                            "blink_rate":
                                self.blink_rate,

                            "anomaly_risk":
                                self.anomaly_risk,

                            "fusion_confidence":
                                self.final_confidence,

                            "final_state":
                                self.final_state,

                            "cognitive_assessment":
                                self.cognitive_assessment
                        })

                    # Run emotion detection every 30 frames
                    if self.frame_count % 30 == 0:

                        emotion_data = self.emotion_detector.analyze_emotion(frame)

                        self.current_emotion = emotion_data["dominant_emotion"]

                        self.blink_rate = (
                            self.visual_analyzer.calculate_blink_rate()
                        )

                        self.stress_level = (
                            self.visual_analyzer.estimate_stress_level()
                        )                       

                    features = {

                        "avg_ear": ear,

                        "avg_yaw": yaw,

                        "avg_pitch": pitch,

                        "avg_roll": roll,

                        "left_gaze_ratio": 0.0,

                        "right_gaze_ratio": 0.0,

                        "center_gaze_ratio": 1.0,

                        "blink_rate": self.blink_rate,

                        "gaze_stability": 1.0,

                        "movement_score": abs(yaw) + abs(pitch)
                    }

                    prediction = self.visual_inference.predict(
                        features
                    )

                    self.predicted_emotion = prediction["emotion"]

                    self.prediction_confidence = (
                        prediction["confidence"] * 100
                    )

                    cv2.putText(
                        frame,
                        f"EAR: {ear:.2f}",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Blinks: {total_blinks}",
                        (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Emotion: {self.current_emotion}",
                        (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 0, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Blink Rate: {self.blink_rate:.1f}/min",
                        (30, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 200, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Stress: {self.stress_level}",
                        (30, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Gaze: {gaze}",
                        (30, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Yaw: {yaw:.1f}",
                        (30, 280),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Pitch: {pitch:.1f}",
                        (30, 320),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Roll: {roll:.1f}",
                        (30, 360),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Cognitive Load: {cognitive_load}",
                        (30, 400),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Deception Risk: {deception_risk}",
                        (30, 440),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 100, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"AI Prediction: {self.predicted_emotion}",
                        (30, 480),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Confidence: {self.prediction_confidence:.1f}%",
                        (30, 520),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Temporal Emotion: "
                        f"{self.temporal_prediction}",
                        (500, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 100, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Temporal Confidence: "
                        f"{self.temporal_confidence:.1f}%",
                        (500, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 100, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Final State:",
                        (500, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        self.final_state,
                        (500, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Cognitive State:",
                        (500, 220),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        self.cognitive_assessment,
                        (500, 260),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Anomaly Risk: "
                        f"{self.anomaly_risk}",
                        (500, 320),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 120, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Fusion Confidence: "
                        f"{self.final_confidence:.2f}",
                        (500, 360),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Risk Level: "
                        f"{self.risk_level}",
                        (500, 400),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Risk Score: "
                        f"{self.risk_score}",
                        (500, 440),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

            frame = self.facemesh.draw_landmarks(frame, results)


            cv2.imshow(
                "MaCoDeR Face Detection",
                frame
            )

            key = cv2.waitKey(1)

            if key & 0xFF == ord('q'):
                break

        self.stop_camera()

    def stop_camera(self):

        if self.cap:
            self.cap.release()

        cv2.destroyAllWindows()

        report = self.report_generator.generate(
            self.session_logger.file_path
        )

        print("\n===== SESSION REPORT =====\n")

        for key, value in report.items():

            print(f"{key}: {value}")

        print("Camera stopped")