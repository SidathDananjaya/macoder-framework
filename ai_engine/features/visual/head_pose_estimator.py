import cv2
import numpy as np
import mediapipe as mp


class HeadPoseEstimator:

    def __init__(self):

        self.face_mesh = mp.solutions.face_mesh

    def estimate_pose(self, frame, face_landmarks):

        img_h, img_w, _ = frame.shape

        # Selected landmark points
        face_2d = []
        face_3d = []

        landmark_ids = [33, 263, 1, 61, 291, 199]

        for idx in landmark_ids:

            lm = face_landmarks.landmark[idx]

            x = int(lm.x * img_w)
            y = int(lm.y * img_h)

            face_2d.append([x, y])

            face_3d.append([x, y, lm.z])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        # Camera matrix
        focal_length = img_w

        cam_matrix = np.array([
            [focal_length, 0, img_h / 2],
            [0, focal_length, img_w / 2],
            [0, 0, 1]
        ])

        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(
            face_3d,
            face_2d,
            cam_matrix,
            dist_matrix
        )

        rmat, jac = cv2.Rodrigues(rot_vec)

        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        pitch = angles[0] * 360
        yaw = angles[1] * 360
        roll = angles[2] * 360

        return {
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll
        }