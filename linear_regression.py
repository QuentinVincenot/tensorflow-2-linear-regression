import tensorflow as tf

class LinearRegression():
  def __init__(self):
    self.slope = tf.Variable(5.0)
    self.intercept = tf.Variable(0.0)

  def __call__(self, X):
    return self.slope * X + self.intercept

  def loss_value(self, prediction, reality):
    return tf.reduce_mean(tf.square(prediction - reality))
