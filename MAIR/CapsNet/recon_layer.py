import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models



class ReconLayer(tf.keras.Model):
    def __init__(self, in_capsule, in_dim, name="", out_dim=28, img_dim=1):
        super(ReconLayer, self).__init__(name=name)

        self.in_capsule = in_capsule
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.y = None

        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(512, name="fc1", activation=tf.nn.relu)
        self.fc2 = layers.Dense(1024, name="fc2", activation=tf.nn.relu)
        self.fc3 = layers.Dense(out_dim * out_dim * img_dim, name="fc3", activation=tf.sigmoid)

    def call(self, x, y):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x