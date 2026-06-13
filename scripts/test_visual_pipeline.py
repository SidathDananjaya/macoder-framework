from ai_engine.features.visual.video_processor import VideoProcessor


def main():

    processor = VideoProcessor()

    processor.start_camera()

    processor.process_stream()


if __name__ == "__main__":
    main()