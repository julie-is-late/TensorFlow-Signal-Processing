import tensorflow as tf

from lowpass import lowpass
from runner import run

def gen_conv(layer_width, filter_size):
    std = 0.1
    alpha = 0.00001

    input_set, output_set, valid_in_batches, valid_out_batches, train_ref_std = lowpass()

    # reshape with channels
    input_set = input_set.reshape(-1, input_set.shape[1], 1)
    output_set = output_set.reshape(-1, output_set.shape[1], 1)
    valid_in_batches = valid_in_batches.reshape(-1, valid_in_batches.shape[1], 1)
    valid_out_batches = valid_out_batches.reshape(-1, valid_out_batches.shape[1], 1)


    ### GEN LAYERS
    x = tf.placeholder(tf.float32, shape=[None, input_set.shape[1], 1], name='x')
    x = tf.expand_dims(x, 1)
    y = tf.placeholder(tf.float32, shape=[None, output_set.shape[1], 1], name='y')
    y = tf.expand_dims(y, 1)

    w0 = tf.Variable(tf.truncated_normal([1, filter_size, 1, layer_width], stddev=std), name='w0')
    b0 = tf.Variable(tf.truncated_normal([layer_width], stddev=std), name='b0')
    conv_0 = tf.nn.conv2d(
        x,
        w0,
        strides=[1,1,1,1],
        padding='SAME')
    lay0 = conv_0 + b0
    lay0 = tf.nn.relu(lay0)

    w1 = tf.Variable(tf.truncated_normal([layer_width], stddev=std), name='w1')
    b1 = tf.Variable(tf.truncated_normal([layer_width], stddev=std), name='b1')
    lay1 = lay0 * w1 + b1
    lay1 = tf.nn.relu(lay1)

    # required b/c conv2d_transpose does not infer None sized object's sizes at runtime, but we can cheat like this
    dyn_input_shape = tf.shape(x)
    batch_size = dyn_input_shape[0]

    w2 = tf.Variable(tf.truncated_normal([1, filter_size, 1, layer_width], stddev=std), name='w2')
    b2 = tf.Variable(tf.truncated_normal([1, 1], stddev=std), name='b2')
    conv_2 = tf.nn.conv2d_transpose(
        lay1,
        w2,
        output_shape=tf.pack([batch_size, 1, output_set.shape[1], 1]),
        strides=[1,1,1,1],
        padding='SAME')
    lay2 = conv_2 + b2


    P = tf.squeeze(lay2) # drop size 1 dim (channels)

    MSE = tf.reduce_mean(tf.square(lay2 - y))
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


def run_conv(hidden_width, filter_size, epochs):
    # oh god what have I done
    x, y, MSE, P, optimizer, global_step, saver, input_set, output_set, valid_in_batches, valid_out_batches, train_ref_std = gen_conv(hidden_width, filter_size)
    run(x, y, MSE, P, optimizer, global_step, saver, input_set, output_set, valid_in_batches, valid_out_batches, train_ref_std, 'lowpass', 'convolution', hidden_width, epochs, filter_size)
