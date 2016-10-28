import tensorflow as tf

from util import weight_variable


def reg_nn(height_in, height_out, node_count):
    """Assumes its fully connected
    Stands for `regression` not `regular`, lol"""

    x = tf.placeholder(tf.float32, shape=[None, height_in])
    y = tf.placeholder(tf.float32, shape=[None, height_out])

    w1 = weight_variable([height_in, node_count])
    b1 = weight_variable([1, node_count])

    w2 = weight_variable([node_count,node_count])
    b2 = weight_variable([1, node_count])

    w3 = weight_variable([node_count,height_out])
    b3 = weight_variable([1, height_out])

    layer1 = tf.add(tf.matmul(x, w1), b1)
    layer1 = tf.nn.relu(layer1)

    layer2 = tf.add(tf.matmul(layer1, w2), b2)
    layer2 = tf.nn.relu(layer2)

    layer3 = tf.add(tf.matmul(layer2, w3), b3)

    ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(layer3, y))
    optimizer = tf.train.AdamOptimizer().minimize(ce)
    y_pred = tf.nn.softmax(layer3)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    return sess, x, y, ce, optimizer, y_pred
