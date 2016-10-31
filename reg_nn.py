import tensorflow as tf
import numpy as np

from util import weight_variable


def gen_layer(input_x, feature_count, node_count, std):
    w = weight_variable([feature_count,node_count], std=std)
    b = weight_variable([1,node_count], std=std)
    return [tf.matmul(input_x,w) + b, tf.nn.l2_loss(w) + tf.nn.l2_loss(b)]

def gen_relu_layer(input_x, feature_count, node_count, std):
    k = gen_layer(input_x, feature_count, node_count, std)
    return [tf.nn.relu(k[0]), k[1]]

def reg_nn(height_in, height_out, hidden_layer_count, node_count, std=0.1, alpha=0.00001):
    """Assumes its fully connected
    Stands for `regression` not `regular`, lol"""

    x = tf.placeholder(tf.float32, shape=[None, height_in])
    y = tf.placeholder(tf.float32, shape=[None, height_out])

    layers = []
    layers.append(gen_relu_layer(x, height_in, node_count, std))

    for _ in range(hidden_layer_count - 1):
        layers.append(gen_relu_layer(layers[-1][0], int(layers[-1][0].get_shape()[1]), node_count, std))

    layers.append(gen_layer(layers[-1][0], int(layers[-1][0].get_shape()[1]), height_out, std))

    layers = np.asarray(layers)

    CE = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(layers[-1][0], y))
    L2 = alpha * (layers[:,1].sum())

    optimizer = tf.train.AdamOptimizer().minimize(CE + L2)

    P = tf.nn.softmax(layers[-1][0])

    # initialization of variables
    init = tf.initialize_all_variables()

    # initialize a computation session
    sess = tf.Session()
    sess.run(init)

    return sess, optimizer, x, y, P, CE
