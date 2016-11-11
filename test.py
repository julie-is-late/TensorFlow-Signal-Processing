import numpy as np

from audio_batch import batch_audio
from convert_data import write_output

def get_test():
    beet_pre = np.load('./data/lowpass/pre/beethoven_opus10_1.npz')['data']
    beet_post = np.load('./data/lowpass/post/beethoven_opus10_1.npz')['data']

    set_in = np.concatenate([beet_pre[0], beet_pre[1]])
    set_out = np.concatenate([beet_post[0], beet_post[1]])

    input_set, output_set = batch_audio(set_in, set_out, .5, offset=.5)

    return input_set, output_set


def run_test(x, y, P, MSE, sess, run_name=None):
    test_input_batched, test_output_batched = get_test()

    if len(x.get_shape()) == 3:
        (test_p, mse) = sess.run([P, MSE],feed_dict={x:test_input_batched.reshape(-1, test_input_batched.shape[1], 1), y:test_output_batched.reshape(-1, test_output_batched.shape[1], 1)})
    else:
        (test_p, mse) = sess.run([P, MSE],feed_dict={x:test_input_batched, y:test_output_batched})

    p = np.squeeze(test_p)

    std = (p - test_output_batched[:,:p.shape[1]]).std()
    print(' test mse:', mse)
    print('test rmse:', np.sqrt(mse))
    print(' test std:', std)

    p = p.reshape(p.shape[0] * p.shape[1])

    filename = 'beethoven_opus10_generated.wav'

    if run_name is not None:
        filename = run_name + '_' + filename

    write_output(p, filename)
