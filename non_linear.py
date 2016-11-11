import tensorflow as tf

from lowpass import lowpass
from runner import run

def gen_nonlin(layer_width):
    std = 0.1
    alpha = 0.00001

    input_set, output_set, valid_in_batches, valid_out_batches, train_ref_std = lowpass()

    ### GEN LAYERS
    x = tf.placeholder(tf.float32, shape=[None, input_set.shape[1]], name='x')
    y = tf.placeholder(tf.float32, shape=[None, output_set.shape[1]], name='y')

    w0 = tf.Variable(tf.truncated_normal([input_set.shape[1], layer_width], stddev=std), name='w0')
    b0 = tf.Variable(tf.truncated_normal([1,layer_width], stddev=std), name='b0')
    lay0 = tf.matmul(x,w0) + b0
    lay0 = tf.nn.relu(lay0)

    w1 = tf.Variable(tf.truncated_normal([layer_width, layer_width], stddev=std), name='w1')
    b1 = tf.Variable(tf.truncated_normal([1,layer_width], stddev=std), name='b1')
    lay1 = tf.matmul(lay0,w1) + b1
    lay1 = tf.nn.relu(lay1)

    w2 = tf.Variable(tf.truncated_normal([layer_width, output_set.shape[1]], stddev=std), name='w2')
    b2 = tf.Variable(tf.truncated_normal([1,output_set.shape[1]], stddev=std), name='b2')
    lay2 = tf.matmul(lay1,w2) + b2

    P = lay2

    MSE = tf.reduce_mean(tf.square(P - y))
    L2 = alpha * (tf.nn.l2_loss(w0) + tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2))

    optimizer = tf.train.AdamOptimizer().minimize(MSE + L2)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    saver = tf.train.Saver(
        { "w0": w0,
          "b0": b0,
          "w1": w1,
          "b1": b1,
          "w2": w2,
          "b2": b2,
          "global_step": global_step})

    return x, y, MSE, P, optimizer, global_step, saver, input_set, output_set, valid_in_batches, valid_out_batches, train_ref_std


def run_nonlin(hidden_width, epochs):
    # oh god what have I done
    x, y, MSE, P, optimizer, global_step, saver, input_set, output_set, valid_in_batches, valid_out_batches, train_ref_std = gen_nonlin(hidden_width)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    run(sess, x, y, MSE, P, optimizer, global_step, saver, input_set, output_set, valid_in_batches, valid_out_batches, train_ref_std, 'lowpass', 'nonlinear', hidden_width, epochs)
    return x, P, sess
