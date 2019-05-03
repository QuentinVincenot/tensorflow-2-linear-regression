import tensorflow as tf

class LinearRegression():
  def __init__(self):
    self.slope = tf.Variable(5.0)
    self.intercept = tf.Variable(0.0)

  def __call__(self, X):
    return self.slope * X + self.intercept

  def train(self, X, Y, epochs=10):
    for epoch in range(epochs):
      current_loss = model.loss_value(model(X_train), Y_train)
      model.train_step(X_train, Y_train)
  
  def train_step(self, X, Y):
    with tf.GradientTape() as tape:
      current_loss = self.loss_value(self(X), Y)
      gradient_slope, gradient_intercept = tape.gradient(current_loss, [self.slope, self.intercept])
      self.slope.assign_sub(self.learning_rate * gradient_slope)
      self.intercept.assign_sub(self.learning_rate * gradient_intercept)

  def loss_value(self, prediction, reality):
    return tf.reduce_mean(tf.square(prediction - reality))
