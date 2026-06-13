from ai_engine.features.audio.audio_preprocessor import (
    load_audio,
    normalize_audio
)

from ai_engine.features.audio.speech_rate import (
    estimate_speech_rate
)

from ai_engine.features.audio.pause_detection import (
    calculate_silence_ratio
)

from ai_engine.features.audio.voice_stability import (
    calculate_voice_stability
)

from ai_engine.features.audio.spectral_features import (
    extract_spectral_centroid,
    extract_zero_crossing_rate
)

AUDIO_PATH = (
    "datasets/raw/RAVDESS/"
    "Audio_Speech_Actors_01-24/"
    "Actor_01/"
    "03-01-05-01-01-01-01.wav"
)

signal, sr = load_audio(AUDIO_PATH)

signal = normalize_audio(signal)

tempo = estimate_speech_rate(signal, sr)

silence = calculate_silence_ratio(signal)

stability = calculate_voice_stability(signal)

centroid = extract_spectral_centroid(signal, sr)

zcr = extract_zero_crossing_rate(signal)

print("Speech Rate:", tempo)

print("Silence Ratio:", silence)

print("Voice Stability:", stability)

print("Spectral Centroid:", centroid)

print("Zero Crossing Rate:", zcr)