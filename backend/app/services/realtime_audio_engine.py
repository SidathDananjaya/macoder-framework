from collections import deque

import librosa
import numpy as np

from ai_engine.features.audio.realtime_audio_processor import (
    RealtimeAudioProcessor
)

audio_processor = RealtimeAudioProcessor()

TARGET_SR = 22050
WINDOW_SECONDS = 3.0
MIN_SECONDS = 1.0
INFER_EVERY = 10

DEFAULT_RESULT = {
    "audio_emotion": "neutral",
    "audio_confidence": 0.0,
    "audio_stress": "LOW",
    "audio_stress_confidence": 0.0,
    "speech_intensity": 0.0,
    "audio_quality": 0.0,
    "response_latency": 0.0,
    "pause_ratio": 0.0,
    "speech_rate": 0.0,
    "voiced_ratio": 0.0,
    "mean_pause": 0.0,
}

_window = deque()
_native_sr = None
_last_result = dict(DEFAULT_RESULT)
_count = 0


def reset_audio():
    global _window, _native_sr, _last_result, _count

    _window = deque()
    _native_sr = None
    _last_result = dict(DEFAULT_RESULT)
    _count = 0


def _ensure_window(sample_rate):
    global _window, _native_sr

    if _native_sr != sample_rate:
        _native_sr = sample_rate
        _window = deque(
            _window,
            maxlen=int(WINDOW_SECONDS * sample_rate),
        )


async def process_audio(audio_data, sample_rate=TARGET_SR):

    global _last_result, _count

    sample_rate = int(sample_rate or TARGET_SR)
    _ensure_window(sample_rate)

    if audio_data:
        _window.extend(audio_data)

    _count += 1

    ready = len(_window) >= int(MIN_SECONDS * sample_rate)

    if ready and _count % INFER_EVERY == 0:
        _last_result = self_infer(sample_rate)

    return _last_result


def self_infer(sample_rate):
    try:
        signal = np.asarray(_window, dtype=np.float32)

        if sample_rate != TARGET_SR:
            signal = librosa.resample(
                signal,
                orig_sr=sample_rate,
                target_sr=TARGET_SR,
            )

        return audio_processor.process_audio_chunk(signal, TARGET_SR)

    except Exception as error:
        print("REALTIME AUDIO ERROR:", error)
        return dict(DEFAULT_RESULT)
