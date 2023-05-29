import numpy as np


def generate_random_dataset(datapoints=250):
    # Generate random values and noise
    X = np.random.randn(datapoints, 1)
    RX = np.random.randn(datapoints, 1)*np.random.randn()
    # Generate random linear-like noisy data
    Y = 2*X + RX
    return X, Y
