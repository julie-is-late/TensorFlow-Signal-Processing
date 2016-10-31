# In[1]:

import time
import numpy as np
from audio_batch import batch_audio, get_valid
from reg_nn import reg_nn


sin_pre = np.load('./data/lowpass/pre/sin.npz')['data']
sqr_pre = np.load('./data/lowpass/pre/square.npz')['data']
saw_pre = np.load('./data/lowpass/pre/plysaw.npz')['data']

sin_post = np.load('./data/lowpass/post/sin.npz')['data']
sqr_post = np.load('./data/lowpass/post/square.npz')['data']
saw_post = np.load('./data/lowpass/post/plysaw.npz')['data']


set_in = np.concatenate([sin_pre[0], sin_pre[1], sqr_pre[0], sqr_pre[1], saw_pre[0], saw_pre[1]])
set_out = np.concatenate([sin_post[0], sin_post[1], sqr_post[0], sqr_post[1], saw_post[0], saw_post[1]])


print('audio shapes should match:', set_in.shape, set_out.shape)


train_in, train_out, valid_in, valid_out = get_valid(set_in, set_out, 1, .25)

if not train_in.shape[0] + valid_in.shape[0] == train_out.shape[0] + valid_out.shape[0] == set_in.shape[0] == set_out.shape[0]:
    raise ValueError('audio shapes don\'t match up')

input_set, output_set = batch_audio(train_in, train_out, .5, offset=.1)

valid_in_batches  = valid_in.reshape(int(valid_in.shape[0] / input_set.shape[1]), input_set.shape[1])
valid_out_batches = valid_out.reshape(int(valid_out.shape[0] / output_set.shape[1]), output_set.shape[1])


# In[ ]:
sess, optimizer, x, y, P, MSE = reg_nn(input_set.shape[1], output_set.shape[1], 3, 1024)

epochs = 500
t_start = time.time()

train_ref = output_set.mean()
valid_ref = valid_out_batches.mean()
train_ref_std = (output_set - train_ref).std()

print('                               mse                                 std            ')
print('    epoch     training  validation    training  validation   reference     runtime')

epoch = 0
for _ in range(epochs):
    perm = np.random.permutation(input_set.shape[0])
    sess.run([optimizer],feed_dict={x:input_set[perm],y:output_set[perm]})

    if epoch % (epochs / 100) == 0:
        (mse_train, p_train) = sess.run([MSE, P],feed_dict={x:input_set,y:output_set})
        (mse_valid, p_valid) = sess.run([MSE, P],feed_dict={x:valid_in_batches,y:valid_out_batches})
        train_std = (output_set - p_train).std()
        valid_std = (valid_out_batches - p_valid).std()

        total_compute_time = (time.time() - t_start)/60
        print('      %3d %12.5f%12.5f%12.5f%12.5f%12.5f%12.1f' % (epoch, mse_train, mse_valid, train_std, valid_std, train_ref_std, total_compute_time))
    epoch+=1

print('\t                       mse                                 std            ')
print('\t      training  validation    training  validation   reference     runtime')
print('epoch:%3d %12.5f%12.5f%12.5f%12.5f%12.5f%12.1f' % (epoch, mse_train, mse_valid, train_std, valid_std, train_ref_std, total_compute_time))


