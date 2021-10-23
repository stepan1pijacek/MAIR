import tensorflow as tf


# TODO: write RELU implementation for NN to evaluate its performance against the squash function

def squash(vectors, axis=-1):
    epsilon = 1e-8
    vector_squared_norm = tf.math.reduce_sum(tf.math.square(vectors), axis, True) + epsilon
    return (vector_squared_norm / (1 + vector_squared_norm)) * (vectors / tf.math.sqrt(vector_squared_norm)) + epsilon
