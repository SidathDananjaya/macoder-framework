import numpy as np


class EyeMetrics:

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    @staticmethod
    def euclidean_distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def calculate_ear(self, eye_points):

        p1, p2, p3, p4, p5, p6 = eye_points

        vertical_1 = self.euclidean_distance(p2, p6)
        vertical_2 = self.euclidean_distance(p3, p5)

        horizontal = self.euclidean_distance(p1, p4)

        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

        return ear

    def extract_eye_points(self, landmarks, eye_indices, width, height):

        points = []

        for idx in eye_indices:
            lm = landmarks[idx]
            x = int(lm.x * width)
            y = int(lm.y * height)

            points.append((x, y))

        return points
    
    def eye_distance(self, left_eye, right_eye):

        left_center = np.mean(left_eye, axis=0)
        right_center = np.mean(right_eye, axis=0)

        return np.linalg.norm(left_center - right_center)

    def get_eye_metrics(self, frame, face_landmarks):

        h, w, _ = frame.shape

        landmarks = face_landmarks.landmark

        left_eye = self.extract_eye_points(
            landmarks,
            self.LEFT_EYE,
            w,
            h
        )

        right_eye = self.extract_eye_points(
            landmarks,
            self.RIGHT_EYE,
            w,
            h
        )

        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)

        avg_ear = (left_ear + right_ear) / 2.0

        eye_width = self.eye_distance(left_eye, right_eye)

        return {
            "left_ear": left_ear,
            "right_ear": right_ear,
            "avg_ear": avg_ear,
            "eye_width": eye_width,
            "left_eye_points": left_eye,
            "right_eye_points": right_eye
        }