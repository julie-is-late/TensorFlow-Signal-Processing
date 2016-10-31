import numpy as np

from util import min_batch

def make_batch(audio_in, audio_out, n, batch_length, sample_offset):
    """This is a tad more complicated than normal b/c batches can overlap"""
    if n * sample_offset > audio_in.shape[0] - batch_length:
        raise ValueError('too many batches %f %f' % (n * sample_offset, audio_in.shape[0] - batch_length))

    if audio_out.shape[0] != audio_in.shape[0]:
        raise ValueError('audio in and audio out are not the same length')

    # perm = np.arange(n)
    input_set = np.zeros([n, 2, batch_length])
    output_set = np.zeros([n, 2, batch_length])

    offset = 0
    for i in range(n):
        input_set[i]  = audio_in[offset:offset + batch_length]
        output_set[i] = audio_out[offset:offset + batch_length]
        offset += sample_offset

    return input_set, output_set


def batch_audio(audio_in, audio_out, seconds, offset=None):
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


    ### calculate number of slices
    n = int((audio_in.shape[0] - sample_length) / sample_offset)

    return make_batch(audio_in, audio_out, n, sample_length, sample_offset)



def get_valid(audio_in, audio_out, seconds, valid_percent):
    """Extracts a validation set from the audio data"""
    # assume 44.1khz wav file
    sample_length = int(44100 * seconds)

    input_set = np.copy(audio_in)
    output_set = np.copy(audio_out)

    # ### Cut out long periods of blank
    # FIXME: this has bugs b/c `input_set[:start]` doesn't fix the offset of what's removed
    # start = None
    # # seems space efficient /s
    # for i in range(audio_in.shape[1]):
    #     if (audio_in[0,i] == 0) and (audio_in[1,i] == 0) and (audio_out[0,i] == 0) and (audio_out[1,i] == 0):
    #         if start is None:
    #             start = i
    #     else:
    #         if start is not None and start - i > 1:
    #             input_set = np.append(input_set[:start], input_set[i:])
    #             output_set = np.append(output_set[:start], output_set[i:])

    ### split off testing set
    num_descrete = int((input_set.shape[0] - sample_length) / sample_length)
    if valid_percent >= 1:
        raise ValueError('Invalid sample validation percentage')
    num_valid = num_descrete * valid_percent
    if num_valid <= 0:
        raise ValueError('Invalid sample validation percentage for given audio sample')

    validation_pos = int((input_set.shape[0] - sample_length * num_valid) / sample_length)
    ix = np.random.permutation(validation_pos)

    valid_set = np.asarray(
        [ input_set[int(ix[0] * sample_length) : int(ix[0] * sample_length + num_valid * sample_length)],
          output_set[int(ix[0] * sample_length) : int(ix[0] * sample_length + num_valid * sample_length)] ])
    input_set = np.concatenate(
        [ input_set[:int(ix[0] * sample_length)],
          input_set[int(ix[0] * sample_length + num_valid * sample_length):] ])
    output_set = np.concatenate(
        [ output_set[:int(ix[0] * sample_length)],
          output_set[int(ix[0] * sample_length + num_valid * sample_length):] ])

    return input_set, output_set, valid_set[0], valid_set[1]

