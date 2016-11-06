import numpy as np
import math
import tensorflow as tf


def min_batch(batch_size, n):
    """min_batch generates a permutation of n elements with a width of batch_size"""
    ix = np.random.permutation(n)
    k = np.empty([math.ceil(float(n) / batch_size)], dtype=object)
    for y in range(0, math.ceil(n / batch_size)):
        k[y] = np.array([], dtype=int)
        for z in range(0, batch_size):
            if y * batch_size + z > n - 1:
                break
            k[y] = np.append(k[y], ix[y * batch_size + z])
    return k


def weight_variable(shape, std=0.1):
    initial = tf.truncated_normal(shape, stddev=std)
    return tf.Variable(initial)

def header():
    print('\t                         mse                    rmse                                 std            ')
    print('\t        training  validation    training  validation    training  validation   reference     runtime')
