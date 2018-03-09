from read_data import *

#################### "read_data" imports:
#
#   all_files = [h5py.File(m, 'r') for m in mat_names]
#   all_ims = [f['image'] for f in all_files]
#   all_types = [f['type'] for f in all_files]
#
####################

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

dtype = torch.FloatTensor # torch.cuda.FloatTensor 

class Flatten(nn.Module):
    def forward(self, x):
        N, num_electrodes, num_times
        return
