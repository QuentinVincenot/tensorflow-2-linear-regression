import numpy as np

def generate_random_dataset(datapoints=250):
	X = np.random.randn(datapoints, 1)
	RX = np.random.randn(datapoints, 1)*np.random.randn()
	Y = 2*X + RX
	return X, Y
