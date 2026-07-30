import librosa
import numpy as np


def extract_mfcc(signal, sr, n_mfcc=13):

    mfccs = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=n_mfcc
    )

    return np.mean(
        mfccs.T,
        axis=0
    )


def extract_pitch(signal, sr):

    pitches, magnitudes = librosa.piptrack(
        y=signal,
        sr=sr
    )

    pitch_values = []

    for i in range(pitches.shape[0]):

        idx = magnitudes[i].argmax()

        pitch = pitches[i, idx]

        if 80 <= pitch <= 400:

            pitch_values.append(
                pitch
            )

    if len(pitch_values) > 0:

        return float(
            np.mean(pitch_values)
        )

    return 0.0


def extract_energy(signal):

    return float(
        np.mean(signal ** 2)
    )


class AudioFeatureExtractor:
    N_MFCC = 13

    def extract_features(self, filepath):

        signal, sr = librosa.load(filepath, sr=22050)

        return self.extract_from_signal(signal, sr)

    def extract_from_signal(self, signal, sr=22050):
        signal = np.asarray(signal, dtype=np.float32)

        features = {}

        if signal.size < 512:
            return self._zero_vector()

        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=self.N_MFCC)
        mfcc_delta = librosa.feature.delta(mfccs)
        chroma = librosa.feature.chroma_stft(y=signal, sr=sr)

        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        dmfcc_mean = np.mean(mfcc_delta, axis=1)
        chroma_mean = np.mean(chroma, axis=1)

        for i in range(self.N_MFCC):
            features[f"mfcc_{i}_mean"] = float(mfcc_mean[i])
        for i in range(self.N_MFCC):
            features[f"mfcc_{i}_std"] = float(mfcc_std[i])
        for i in range(self.N_MFCC):
            features[f"dmfcc_{i}_mean"] = float(dmfcc_mean[i])
        for i in range(chroma.shape[0]):
            features[f"chroma_{i}"] = float(chroma_mean[i])

        features["zcr"] = float(
            np.mean(librosa.feature.zero_crossing_rate(signal))
        )
        features["rms"] = float(np.mean(librosa.feature.rms(y=signal)))
        features["spectral_centroid"] = float(
            np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr))
        )
        features["spectral_bandwidth"] = float(
            np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sr))
        )
        features["spectral_rolloff"] = float(
            np.mean(librosa.feature.spectral_rolloff(y=signal, sr=sr))
        )

        return features

    def feature_names(self):
        names = (
            [f"mfcc_{i}_mean" for i in range(self.N_MFCC)]
            + [f"mfcc_{i}_std" for i in range(self.N_MFCC)]
            + [f"dmfcc_{i}_mean" for i in range(self.N_MFCC)]
            + [f"chroma_{i}" for i in range(12)]
            + ["zcr", "rms", "spectral_centroid",
               "spectral_bandwidth", "spectral_rolloff"]
        )
        return names

    def _zero_vector(self):
        return {name: 0.0 for name in self.feature_names()}
