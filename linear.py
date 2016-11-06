import time, os, math

import numpy as np
import tensorflow as tf

from lowpass import lowpass
from util import header

def gen_lin(layer_width):
    std = 0.1
    alpha = 0.00001

    input_set, output_set, valid_in_batches, valid_out_batches, train_ref_std = lowpass()

    ### GEN LAYERS
    x = tf.placeholder(tf.float32, shape=[None, input_set.shape[1]], name='x')
    y = tf.placeholder(tf.float32, shape=[None, output_set.shape[1]], name='y')

    w0 = tf.Variable(tf.truncated_normal([input_set.shape[1], layer_width], stddev=std), name='w0')
    b0 = tf.Variable(tf.truncated_normal([1,layer_width], stddev=std), name='b0')
    lay0 = tf.matmul(x,w0) + b0

    w1 = tf.Variable(tf.truncated_normal([layer_width, layer_width], stddev=std), name='w1')
    b1 = tf.Variable(tf.truncated_normal([1,layer_width], stddev=std), name='b1')
    lay1 = tf.matmul(lay0,w1) + b1

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


def run_lin(hidden_width, epochs):
    # oh god what have I done
    x, y, MSE, P, optimizer, global_step, saver, input_set, output_set, valid_in_batches, valid_out_batches, train_ref_std = gen_lin(hidden_width)

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    ckpt_dir = "./tmp/lowpass/linear/%d/" % hidden_width

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    else:
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('restoring network from:',ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

    batch_size = 500
    epoch = sess.run(global_step)
    t_start = time.time()
    total_compute_time = -1

    check_dist = int((epochs / 100))

    print("starting from epoch:", epoch)

    if epoch < epochs:
        header()

    while epoch < epochs:
        perm = np.random.permutation(input_set.shape[0])

        start = 0
        for _ in range( math.ceil( input_set.shape[0] / batch_size ) ):
            batch = perm[ start:start + batch_size ]
            sess.run([optimizer],feed_dict={x:input_set[batch],y:output_set[batch]})
            start += batch_size
        print('.', end="", flush=True)

        epoch+=1
        sess.run(global_step.assign(epoch))

        if epoch % check_dist == 0:
            saver.save(sess, ckpt_dir + 'model.ckpt')
            (mse_train, p_train) = sess.run([MSE, P],feed_dict={x:input_set,y:output_set})
            (mse_valid, p_valid) = sess.run([MSE, P],feed_dict={x:valid_in_batches,y:valid_out_batches})
            train_std = (output_set - p_train).std()
            valid_std = (valid_out_batches - p_valid).std()

            total_compute_time = (time.time() - t_start)/60
            print()
            print('epoch:%5d %12.5f%12.5f%12.5f%12.5f%12.5f%12.5f%12.5f%12.1f' % (epoch, mse_train, mse_valid, np.sqrt(mse_train), np.sqrt(mse_valid), train_std, valid_std, train_ref_std, total_compute_time), end=" ")


    # compute final results (and ensure computed if we're already done)
    (mse_train, p_train) = sess.run([MSE, P],feed_dict={x:input_set,y:output_set})
    (mse_valid, p_valid) = sess.run([MSE, P],feed_dict={x:valid_in_batches,y:valid_out_batches})
    train_std = (output_set - p_train).std()
    valid_std = (valid_out_batches - p_valid).std()

    print()
    header()
    print('epoch:%5d %12.5f%12.5f%12.5f%12.5f%12.5f%12.5f%12.5f%12.1f' % (epoch, mse_train, mse_valid, np.sqrt(mse_train), np.sqrt(mse_valid), train_std, valid_std, train_ref_std, total_compute_time))
