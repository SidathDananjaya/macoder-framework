from pathlib import Path


class RavdessVideoLoader:

    def __init__(self, dataset_path):

        self.dataset_path = Path(dataset_path)

    def get_video_files(self):

        video_files = []

        for actor_dir in self.dataset_path.iterdir():

            if actor_dir.is_dir():

                for video_file in actor_dir.glob("*.mp4"):

                    video_files.append(video_file)

        return video_files