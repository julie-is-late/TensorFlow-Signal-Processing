#!/usr/bin/env python
"""This script converts wav files from 8, 16, or 24 bit PCM(Integer) to np.float32 np.arrays
    and saves them to the right locations"""

import os

import numpy as np

import modules.wavio as wavio


def export_data(directory, filename):
    """Converts wav files from 8, 16, or 24 bit PCM(Integer) to np.float32 np.arrays and saves them to the right locations based on filename.

    Args:
        directory (str): relative directory the file is in.
        filename (str): must be of the format "<filter> - <pre/post> - <sound name>.wav".

    Returns:
        nothing
    """

    OUT_DIR = './data'

    inn = wavio.read(os.path.join(directory, filename))

    # convert to float and normalize if needed
    # help from https://github.com/mgeier/python-audio/blob/master/audio-files/utility.py
    if inn.data.dtype.kind in 'iu':
        i = np.iinfo(inn.data.dtype)
        abs_max = 2 ** (i.bits - 1)
        offset = i.min + abs_max
        floats = (inn.data.astype(np.float32) - offset) / abs_max

    split = filename.split(" - ")

    filter_dir = split[0]
    pre_post = split[1]
    save_file_name = split[2].split(".")[0]

    if not os.path.exists(os.path.join(OUT_DIR, filter_dir)):
        os.makedirs(os.path.join(OUT_DIR, filter_dir))
    if not os.path.exists(os.path.join(OUT_DIR, filter_dir, pre_post)):
        os.makedirs(os.path.join(OUT_DIR, filter_dir, pre_post))

    np.savez_compressed(os.path.join(OUT_DIR, filter_dir, pre_post, save_file_name), data=np.swapaxes(floats,0,1))


def export_filtered_audio(directory):
    """Exports audio used as our data"""
    # DIRECTORY = './sound_files'

    for file in os.listdir(directory):
        if file.endswith(".wav"):
            export_data(directory, file)
