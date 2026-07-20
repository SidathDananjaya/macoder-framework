from ai_engine.inference.realtime.audio_emotion_inference import (
    AudioEmotionInference
)

model = AudioEmotionInference()

result = model.predict({

    "mfcc_mean": -46.0,

    "mfcc_std": 180.0,

    "zcr": 0.35,

    "rms": 0.004,

    "spectral_centroid": 3100
})

print(result)