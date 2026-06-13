import numpy as np


class TemporalDatasetBuilder:

    def __init__(self):

        self.sequences = []
        self.labels = []

    def add_sample(self, sequence, label):

        self.sequences.append(sequence)
        self.labels.append(label)

    def save(self):

        np.save(
            "temporal_sequences.npy",
            np.array(self.sequences)
        )

        np.save(
            "temporal_labels.npy",
            np.array(self.labels)
        )

        print("Temporal dataset saved")