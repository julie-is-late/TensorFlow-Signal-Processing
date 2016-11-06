import numpy as np

from audio_batch import batch_audio, get_valid


def lowpass():
    sin_pre = np.load('./data/lowpass/pre/sin.npz')['data']
    sqr_pre = np.load('./data/lowpass/pre/square.npz')['data']
    saw_pre = np.load('./data/lowpass/pre/plysaw.npz')['data']

    sin_post = np.load('./data/lowpass/post/sin.npz')['data']
    sqr_post = np.load('./data/lowpass/post/square.npz')['data']
    saw_post = np.load('./data/lowpass/post/plysaw.npz')['data']


    set_in = np.concatenate([sin_pre[0], sin_pre[1], sqr_pre[0], sqr_pre[1], saw_pre[0], saw_pre[1]])
    set_out = np.concatenate([sin_post[0], sin_post[1], sqr_post[0], sqr_post[1], saw_post[0], saw_post[1]])

    train_in, train_out, valid_in, valid_out = get_valid(set_in, set_out, 1, .25)

    if not train_in.shape[0] + valid_in.shape[0] == train_out.shape[0] + valid_out.shape[0] == set_in.shape[0] == set_out.shape[0]:
        raise ValueError('audio shapes don\'t match up')

    input_set, output_set = batch_audio(train_in, train_out, .5, offset=.1)
    valid_in_batches  = valid_in.reshape(int(valid_in.shape[0] / input_set.shape[1]), input_set.shape[1])
    valid_out_batches = valid_out.reshape(int(valid_out.shape[0] / output_set.shape[1]), output_set.shape[1])

    train_ref_std = output_set.std()

    # input_set = input_set.reshape(-1, input_set.shape[1], 1)
    # output_set = output_set.reshape(-1, output_set.shape[1], 1)
    # valid_in_batches = valid_in_batches.reshape(-1, valid_in_batches.shape[1], 1)
    # valid_out_batches = valid_out_batches.reshape(-1, valid_out_batches.shape[1], 1)
    return input_set, output_set, valid_in_batches, valid_out_batches, train_ref_std
