import time, os, math

import numpy as np
import tensorflow as tf

from util import header

def run(sess, x, y, MSE, P, optimizer, global_step, run_time, saver, input_set, output_set, valid_in_batches, valid_out_batches, train_ref_std, dataset, net_type, hidden_width, epochs, batch_size=500, extra=None, check_dist=None):
    try:
        actually_run(sess, x, y, MSE, P, optimizer, global_step, run_time, saver, input_set, output_set, valid_in_batches, valid_out_batches, train_ref_std, dataset, net_type, hidden_width, epochs, batch_size=batch_size, extra=extra, check_dist=check_dist)
    except KeyboardInterrupt:
        print('Interrupted')


def actually_run(sess, x, y, MSE, P, optimizer, global_step, run_time, saver, input_set, output_set, valid_in_batches, valid_out_batches, train_ref_std, dataset, net_type, hidden_width, epochs, batch_size=500, extra=None, check_dist=None):
    ckpt_dir = "./tmp/%s/%s/%d/" % (dataset, net_type, hidden_width)
    if extra is not None:
        ckpt_dir += '%d/' % (extra)

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    else:
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('restoring network from:',ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

    epoch = sess.run(global_step)
    total_time = sess.run(run_time)
    run_start = time.time()

    if check_dist is None:
        check_dist = epochs // 100

    print("starting from epoch:", epoch)

    # only print if printing more than once
    printing = False
    if epoch + check_dist < epochs:
        header(newLine=False)
        printing = True

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
            curr_time = time.time()
            total_time += (curr_time - run_start)/60
            sess.run(run_time.assign(total_time))
            run_start = curr_time

            saver.save(sess, ckpt_dir + 'model.ckpt')
            (mse_train, p_train) = sess.run([MSE, P],feed_dict={x:input_set,y:output_set})
            (mse_valid, p_valid) = sess.run([MSE, P],feed_dict={x:valid_in_batches,y:valid_out_batches})
            train_std = (np.squeeze(output_set) - p_train).std()
            valid_std = (np.squeeze(valid_out_batches) - p_valid).std()

            if printing:
                print()
                print('epoch:%5d %12.5f%12.5f%12.5f%12.5f%12.5f%12.5f%12.5f%12.1f' % (epoch, mse_train, mse_valid, np.sqrt(mse_train), np.sqrt(mse_valid), train_std, valid_std, train_ref_std, total_time), end=" ")


    # compute final results (and ensure computed if we're already done)
    (mse_train, p_train) = sess.run([MSE, P],feed_dict={x:input_set,y:output_set})
    (mse_valid, p_valid) = sess.run([MSE, P],feed_dict={x:valid_in_batches,y:valid_out_batches})
    train_std = (np.squeeze(output_set) - p_train).std()
    valid_std = (np.squeeze(valid_out_batches) - p_valid).std()

    print()
    header()
    print('epoch:%5d %12.5f%12.5f%12.5f%12.5f%12.5f%12.5f%12.5f%12.1f' % (epoch, mse_train, mse_valid, np.sqrt(mse_train), np.sqrt(mse_valid), train_std, valid_std, train_ref_std, total_time))

