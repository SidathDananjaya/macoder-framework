import numpy as np


class MultimodalFusion:

    def build_feature_vector(

        self,

        audio_features,
        visual_features,
        behavioral_features,
        temporal_features
    ):

        fused_vector = []

        # Audio
        fused_vector.extend(audio_features)

        # Visual
        fused_vector.extend(visual_features)

        # Behavioral
        fused_vector.extend(behavioral_features)

        # Temporal
        fused_vector.extend(temporal_features)

        return np.array(
            fused_vector,
            dtype=np.float32
        )