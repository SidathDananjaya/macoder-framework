import numpy as np


def calculate_silence_ratio(
    signal,
    threshold=0.01
):

    silent = np.sum(np.abs(signal) < threshold)

    total = len(signal)

    return silent / total