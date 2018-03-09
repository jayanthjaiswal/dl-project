from includes import *

#################### "includes" imports:
#
#   from read_data import *
#   
#   import torch
#   from torch.autograd import Variable
#   import torch.nn as nn
#   import torch.optim as optim
#
#   dtype = torch.cuda.FloatTensor # torch.FloatTensor
#
#   all_files = [h5py.File(m, 'r') for m in mat_names]
#   all_ims = [f['image'] for f in all_files]
#   all_types = [f['type'] for f in all_files]
#
####################

import torch.nn.utils.rnn.pack_padded_sequence as ppseq

TEST_SUBJECT_ID = 0

unpadded_X = dtype(all_ims[TEST_SUBJECT_ID][:,:,:])
Y = variable(dtype(all_types[TEST_SUBJECT_ID][0,:]))

# unpadded_X should have shape (num samples, num_sensors=25, seq_len=1000)
# Y should have shape (num samples)
# in this file I use num samples to mean batch size

N, E, T = unpadded_X.shape
assert N == Y.shape[1]

# for rnn, X should have shape (seq_len=1000, num samples, 25)
swapped_X = Variable(np.swapaxes(unpadded_X, 1, 2).T)

# for generality, we'll pack these into variable length arrays
X = PPSeq(input = swapped_X, lengths = T)
