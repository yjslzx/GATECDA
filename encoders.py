import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import Model

class autoencoder(Model):
  def __init__(self,units):
    super(autoencoder, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(units[0], activation="relu"),
      layers.Dense(units[1], activation="relu"),
      layers.Dense(units[2], activation="relu")])

    self.decoder = tf.keras.Sequential([
      layers.Dense(units[2], activation="relu"),
      layers.Dense(units[1], activation="relu"),
      layers.Dense(units[0], activation="sigmoid")])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

  def infer(self, x):
    return self.encoder(x)