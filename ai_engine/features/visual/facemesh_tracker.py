import cv2
import mediapipe as mp


class FaceMeshTracker:
    def __init__(self,
                 max_faces=1,
                 refine_landmarks=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.mp_draw = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_draw.DrawingSpec(
            thickness=1,
            circle_radius=1
        )

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)

        return results

    def draw_landmarks(self, frame, results):

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                self.mp_draw.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec
                )

        return frame