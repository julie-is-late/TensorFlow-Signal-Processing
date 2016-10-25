import os

import numpy as np

import modules.wavio as wavio

FILENAME = './sound_files/lowpass - post - plysaw.wav'
OUT_DIR = './data'

inn = wavio.read(FILENAME)

# convert to float and normalize if needed
# help from https://github.com/mgeier/python-audio/blob/master/audio-files/utility.py
if inn.data.dtype.kind in 'iu':
    i = np.iinfo(inn.data.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    floats = (inn.data.astype(np.float32) - offset) / abs_max

split = FILENAME.split("/")[2].split(" - ")

filter_dir = split[0]
pre_post = split[1]
save_file_name = split[2].split(".")[0]

if not os.path.exists(os.path.join(OUT_DIR, filter_dir)):
    os.makedirs(os.path.join(OUT_DIR, filter_dir))
if not os.path.exists(os.path.join(OUT_DIR, filter_dir, pre_post)):
    os.makedirs(os.path.join(OUT_DIR, filter_dir, pre_post))

np.savez_compressed(os.path.join(OUT_DIR, filter_dir, pre_post, save_file_name), data=np.swapaxes(floats,0,1))
