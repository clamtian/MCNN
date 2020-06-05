from ops import *
from config import *
import numpy as np

class net:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, train_phase):
        with tf.variable_scope(self.name):
            inputs = featurextract(inputs)
            inputs = tf.nn.relu(conv("conv0", inputs, 64, 3, 1))
            for b in np.arange(0, BLOCK - 1):
                inputs = self.id_block(inputs, str(b), train_phase)
            inputs = conv("last_conv", inputs, IMG_C, 3, 1)
        return inputs

    def id_block(self, inputs, stage, train_phase):
        block_name = "block" + stage
        with tf.variable_scope(block_name):
            out_short = inputs
            outputs = tf.nn.relu(batchnorm(conv(block_name + "_conv0", inputs, 64, 3, 1),
                                          train_phase, block_name + "_bn0"))
            outputs = tf.nn.relu(batchnorm(conv(block_name + "_conv1", outputs, 64, 3, 1),
                                          train_phase, block_name + "_bn1"))
            outputs = tf.nn.relu(batchnorm(conv(block_name + "_conv2", outputs, 64, 3, 1),
                                           train_phase, block_name + "_bn2"))
            add = tf.add(outputs, out_short)
            add_result = tf.nn.relu(add)
        return add_result
