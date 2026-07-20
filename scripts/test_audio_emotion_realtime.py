import numpy as np

from ai_engine.features.audio.realtime_audio_processor import (
    RealtimeAudioProcessor
)

processor = (
    RealtimeAudioProcessor()
)

dummy_audio = np.random.randn(
    22050 * 3
)

result = (
    processor.process_audio_chunk(
        dummy_audio
    )
)

print(result)