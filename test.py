import numpy as np

from audio_batch import batch_audio

def get_test():
    beet_pre = np.load('./data/lowpass/pre/beethoven_opus10_1.npz')['data']
    beet_post = np.load('./data/lowpass/post/beethoven_opus10_1.npz')['data']

    set_in = np.concatenate([beet_pre[0], beet_pre[1]])
    set_out = np.concatenate([beet_post[0], beet_post[1]])

    input_set, output_set = batch_audio(set_in, set_out, .5, offset=.5)

    return input_set, output_set
