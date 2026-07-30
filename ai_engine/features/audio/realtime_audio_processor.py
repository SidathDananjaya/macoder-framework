import tempfile
import numpy as np
import soundfile as sf
import pandas as pd

from backend.app.services.stress_inference_service import (
    predict_stress
)

from ai_engine.features.audio.audio_feature_extractor import (
    AudioFeatureExtractor
)

from ai_engine.inference.realtime.audio_emotion_inference import (
    AudioEmotionInference
)

from ai_engine.features.quality.quality_assessor import (
    assess_audio_quality
)

from ai_engine.features.audio.timing_features import (
    extract_timing_features,
    TIMING_KEYS
)

audio_extractor = AudioFeatureExtractor()

audio_emotion_model = (
    AudioEmotionInference()
)


class RealtimeAudioProcessor:

    def process_audio_chunk(
        self,
        audio_data,
        sample_rate=22050
    ):

        try:

            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False
            ) as temp_file:

                sf.write(
                    temp_file.name,
                    audio_data,
                    sample_rate
                )

                temp_path = temp_file.name


            stress_result = (
                predict_stress(temp_path)
            )


            features = (
                audio_extractor.extract_features(
                    temp_path
                )
            )


            emotion_result = (
                audio_emotion_model.predict(
                    features
                )
            )


            speech_intensity = float(
                np.sqrt(
                    np.mean(
                        np.square(audio_data)
                    )
                )
            ) if audio_data.size else 0.0


            audio_quality = assess_audio_quality(
                audio_data,
                sample_rate
            )


            timing = extract_timing_features(
                audio_data,
                sample_rate
            )

            return {

                **timing,

                "audio_emotion":
                    emotion_result["emotion"],

                "audio_confidence":
                    emotion_result["confidence"],

                "audio_stress":
                    str(stress_result["stress_level"]).upper(),

                "audio_stress_confidence":
                    stress_result["confidence"],

                "speech_intensity":
                    round(speech_intensity, 4),

                "audio_quality":
                    round(audio_quality, 3)
            }

        except Exception as e:

            print(
                "AUDIO PROCESSOR ERROR:",
                e
            )

            return {

                **{k: 0.0 for k in TIMING_KEYS},

                "audio_emotion":
                    "neutral",

                "audio_confidence":
                    0.0,

                "audio_stress":
                    "LOW",

                "audio_stress_confidence":
                    0.0,

                "speech_intensity":
                    0.0,

                "audio_quality":
                    0.0
            }
