import cv2


def extract_frames(
    video_path,
    frame_rate=5
):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        raise Exception(
            f"Cannot open video: {video_path}"
        )

    frames = []

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_interval = int(fps / frame_rate)

    count = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        if count % frame_interval == 0:

            frames.append(frame)

        count += 1

    cap.release()

    return frames