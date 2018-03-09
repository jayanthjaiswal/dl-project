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
from collections import OrderedDict

class VanillaRNN(nn.Module):

    def __init__(self,
            initial_hidden_layer_sizes=[10]):
        super(VanillaRNN, self).__init__()

        # first layers converts input feature vectors
        prev_size = E
        initial_layers = OrderedDict() 
        for idx, s in enumerate(initial_hidden_layer_sizes):
            initial_layers['initial_'+idx] = nn.Linear(prev_size, s)
            prev_size = s

        self.feature_layer = nn.Sequential(initial_layers)

    def forward(self):
        self.
