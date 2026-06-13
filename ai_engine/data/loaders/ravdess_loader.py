import os

from ai_engine.configs.project_paths import RAVDESS_PATH


def load_ravdess_files():

    audio_files = []

    for actor_folder in os.listdir(RAVDESS_PATH):

        actor_path = os.path.join(RAVDESS_PATH, actor_folder)

        if os.path.isdir(actor_path):

            for file in os.listdir(actor_path):

                if file.endswith(".wav"):

                    full_path = os.path.join(actor_path, file)

                    audio_files.append(full_path)

    return audio_files


if __name__ == "__main__":

    files = load_ravdess_files()

    print(f"Total audio files: {len(files)}")
    print(files[:5])