import tensorflow as tf


class LinearRegression():
    def __init__(self):
        # Initialize model hyperparameters with random values at first
        self.slope = tf.Variable(5.0)
        self.intercept = tf.Variable(0.0)
        self.learning_rate = 0.01

    def __call__(self, X):
        # Make a prediction with the model on X input values
        return self.slope * X + self.intercept

    def train(self, X, Y, epochs=10):
        # Train the model on training data for epoch rounds over all dataset
        for epoch in range(epochs):
            self.train_step(X, Y)

    def train_step(self, X, Y):
        with tf.GradientTape() as tape:
            # Compute the loss value with the model on training data
            current_loss = self.loss_value(self(X), Y)
            # Compute gradient for hyperparameters of the model (slope and intercept)
            gradient_slope, gradient_intercept = tape.gradient(
                current_loss, [self.slope, self.intercept])
            # Update current model values by substracting gradients accordingly
            self.slope.assign_sub(self.learning_rate * gradient_slope)
            self.intercept.assign_sub(self.learning_rate * gradient_intercept)

    def loss_value(self, prediction, reality):
        # Compute the mean squared error between prediction and reality
        return tf.reduce_mean(tf.square(prediction - reality))
