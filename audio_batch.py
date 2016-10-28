import numpy as np


def make_batch(audio, n, batch_length, sample_offset):
    """This is a tad more complicated than normal b/c batches can overlap"""
    if n * sample_offset > audio.shape[1] - batch_length:
        raise ValueError('too many batches %f %f' % (n * sample_offset, audio.shape[1] - batch_length))

    # perm = np.arange(n)
    out = np.zeros([n, 2, batch_length])

    offset = 0
    for i in range(n):
        out[i] = audio[:, offset:offset + batch_length]
        offset += sample_offset

    return out

def batch_audio(audio, seconds, offset=None):
    """Automatically batch the audio into sections of length `seconds`"""
    # assume 44.1khz wav file
    sample_length = int(44100 * seconds)

    if offset is None:
        offset = seconds
        print('hello world')
    sample_offset = int(44100 * offset)

    n = int((audio.shape[1] - sample_length) / sample_offset)

    return make_batch(audio, n, sample_length, sample_offset)
