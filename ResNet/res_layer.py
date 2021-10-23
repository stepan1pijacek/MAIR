import tensorflow as tf
from Utils.activation_function import squash

layers = tf.keras.layers
models = tf.keras.models


class Residual(tf.keras.module):
    def call(self, out_prev, out_skip):
        x = tf.keras.layers.Add()([out_prev, out_skip])
        return x

    def count_params(self):
        return 0
    