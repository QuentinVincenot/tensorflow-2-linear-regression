from data_generation import *
from linear_regression import LinearRegression
import matplotlib.pyplot as plt


# Generate training data and test data
X_train, Y_train = generate_random_dataset(datapoints=100)
X_test, _ = generate_random_dataset(datapoints=25)

# Create the LinearRegression model
model = LinearRegression()

# Visualize the model initial performances on training data
limits = [X_train.min()-1, X_train.max()+1]
plt.scatter(X_train, Y_train, c='b')
plt.plot(limits, model(limits))
plt.scatter(X_test, model(X_test), c='y')
plt.show()

# Train the LinearRegression model
model.train(X_train, Y_train, epochs=300)

# Visualize again the model performances on testing data after being trained
plt.scatter(X_train, Y_train, c='b')
plt.plot(limits, model(limits))
plt.scatter(X_test, model(X_test), c='y')
plt.show()
