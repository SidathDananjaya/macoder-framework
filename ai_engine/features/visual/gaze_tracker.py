class GazeTracker:

    def __init__(self):

        self.current_gaze = "CENTER"

    def estimate_gaze(
            self,
            face_landmarks
    ):

        left_iris = face_landmarks.landmark[468]
        right_iris = face_landmarks.landmark[473]

        left_eye_corner = face_landmarks.landmark[33]
        right_eye_corner = face_landmarks.landmark[263]

        avg_iris_x = (
            left_iris.x + right_iris.x
        ) / 2

        avg_eye_x = (
            left_eye_corner.x + right_eye_corner.x
        ) / 2

        diff = avg_iris_x - avg_eye_x

        if diff < -0.02:
            self.current_gaze = "LEFT"

        elif diff > 0.02:
            self.current_gaze = "RIGHT"

        else:
            self.current_gaze = "CENTER"

        return self.current_gaze