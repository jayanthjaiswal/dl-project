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

from collections import OrderedDict

class FC(nn.Module):

    def __init__(self, input_size=22*1000, hidden_layer_sizes=[100], num_classes=4):

        super(FC, self).__init__()

        layers = OrderedDict() 

        prev_size = input_size
        for idx, s in enumerate(hidden_layer_sizes):
            layers['lin_{}'.format(idx)] = nn.Linear(prev_size, s)
            layers['relu_{}'.format(idx)] = nn.ReLU()
            prev_size = s

        layers['output'] = nn.Linear(prev_size, num_classes)

        self.network = nn.Sequential(layers)

    def forward(self, X):
        '''
        X should have shape (num samples, num_sensors=25, seq_len=1000)
        '''
        N, E, T = X.shape

        out = X.view(N, -1)
        out = self.network(out)

        return out
