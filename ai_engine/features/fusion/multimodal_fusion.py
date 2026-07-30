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

        fused_vector.extend(audio_features)

        fused_vector.extend(visual_features)

        fused_vector.extend(behavioral_features)

        fused_vector.extend(temporal_features)

        return np.array(
            fused_vector,
            dtype=np.float32
        )
