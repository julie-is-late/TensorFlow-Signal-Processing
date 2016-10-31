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

def batch_audio(audio, seconds, valid_percent, offset=None):
    """Automatically batch the audio into sections of length `seconds`

    returns batch set and validation set"""

    ### basic time calculations
    # assume 44.1khz wav file
    sample_length = int(44100 * seconds)

    if offset is None:
        # give it some arbitrary separation
        # an offset of 0.1 with 1 second means each value is used in 10 batches
        offset = min(seconds, 0.1)
        print('using an offset of', offset, 'seconds')
    sample_offset = int(44100 * offset)


    ### Cut out long periods of blank
    start = None
    # seems space efficient /s
    input_set = np.copy(audio)
    for i in range(audio.shape[1]):
        if (audio[0,i] == 0) and (audio[1,i] == 0):
            if start is None:
                start = i
        else:
            if start is not None and start - i > 1:
                input_set = np.append(input_set[:start], input_set[i:])

    ### split off testing set
    num_descrete = int((input_set.shape[1] - sample_length) / sample_length)
    if valid_percent >= 1:
        raise ValueError('Invalid sample validation percentage')
    num_valid = num_descrete * valid_percent
    if num_valid <= 0:
        raise ValueError('Invalid sample validation percentage for given audio sample')

    validation_pos = int((input_set.shape[1] - sample_length * num_valid) / sample_length)
    ix = np.random.permutation(validation_pos)

    valid_set = audio[:, int(ix[0] * sample_length) : int(ix[0] * sample_length + num_valid * sample_length)]
    audio = np.concatenate(
        [ audio[:, 0: :int(ix[0] * sample_length)],
          audio[:, int(ix[0] * sample_length + num_valid * sample_length):] ],
        axis=1)

    ### calculate number of slices
    n = int((input_set.shape[1] - sample_length) / sample_offset)

    return make_batch(input_set, n, sample_length, sample_offset), valid_set
