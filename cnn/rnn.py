import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
from collections import OrderedDict
import resource
import pickle
import gc

from parametrized_cnn import *

class Large_RNN(Large_CNN):

    def __init__(self,
        ######################## Parent Large_CNN Params
        # Data
        eeg_channels            = 22,
        input_timesteps         = 1000,

        # Block 1
        conv1_kernel_size       = 10,
        conv1_out_channels      = 25,
        kernels_get_all_channels= False,
            # the paper isn't clear about this, on pg 7 they said each
            # filter gets all channels, but the picture has a 10x1 kernel
        conv2_out_channels      = 25,
        batchnorm2_channels     = True,
        pool2_kernel_size       = 3,
        pool2_stride            = 3,

        # Block 2
        conv3_out_channels      = 50,
        conv3_kernel_size       = 10,
        pool3_kernel_size       = 5, # was 3
        pool3_stride            = 5, # was 3

        # Block 3
        conv4_out_channels      = 100,
        conv4_kernel_size       = 10,
        pool4_kernel_size       = 4, # was 3
        pool4_stride            = 4, # was 3

        # Block 4
        conv5_out_channels      = 200,
        conv5_kernel_size       = 10,
        pool5_kernel_size       = 3,
        pool5_stride            = 3,

        # Classification
        num_classes             = 4,

        # Traininng
        parent_use_dropout      = False,
        parent_verbose          = False
        
        ):

        super(Large_RNN, self).__init__(eeg_channels,
                                        input_timesteps,
                                        conv1_kernel_size,
                                        conv1_out_channels,
                                        kernels_get_all_channels,
                                        conv2_out_channels,
                                        batchnorm2_channels,
                                        pool2_kernel_size,
                                        pool2_stride,
                                        conv3_out_channels,
                                        conv3_kernel_size,
                                        pool3_kernel_size,
                                        pool3_stride,
                                        conv4_out_channels,
                                        conv4_kernel_size,
                                        pool4_kernel_size,
                                        pool4_stride,
                                        conv5_out_channels,
                                        conv5_kernel_size,
                                        pool5_kernel_size,
                                        pool5_stride,
                                        num_classes,
                                        parent_use_dropout,
                                        parent_verbose)


        # after block 1
        #self.rnn_layers = nn.LSTM(  input_size  = 25,
        #                            hidden_size = 200,
        #                            num_layers  = 3,
        #                            batch_first = True)

        # only after block 1 (add classification now)
        #self.classification = nn.Sequential(
        #                        nn.Linear(200, num_classes),
        #                        nn.Softmax(dim=0))

        self.block3 = None
        self.block4 = None
        # after block 2
        self.rnn_layers = nn.LSTM(  input_size  = 50,
                                    hidden_size = 100,
                                    num_layers  = 3,
                                    batch_first = True)

        # only after block 2 (add classification now)
        self.classification = nn.Sequential(
                                nn.Linear(100, num_classes),
                                nn.Softmax(dim=0))

        #self.classification = nn.Sequential(
        #                        nn.Linear(16, num_classes),
        #                        nn.Softmax(dim=0))



    def forward(self, x):
        N, E, T = x.shape

        out = self.norm_layer(x)

        ############ Block 1
        out = out.unsqueeze(1) # to get N, 1, E, T
        out = self.conv1(out)
        out = self.conv2(out)
        out = out.squeeze() # get back to N, E, T

        if (not self.batchnorm2_channels):
            out = self.norm2_opt2(out)

        if (self.batchnorm2_channels):
            out = self.norm2_opt1(out)

        out = self.elu2(out)
        out = self.maxpool2(out)

        if (self.use_dropout):
            out = self.dropouts[0](out)

        ############# Block 2
        out = self.block2(out)

        if (self.use_dropout):
            out = self.dropouts[1](out)

        out = out.permute(0, 2, 1)
        out, state = self.rnn_layers(out)
        #out = out.permute(0, 2, 1)
        h, c = state
        out = h[-1]

        ############# Block 3
        #out = self.block3(out)

        #if (self.use_dropout):
        #    out = self.dropouts[2](out)


        ############# Block 4
        #out = self.block4(out)

        #if (self.use_dropout):
        #    out = self.dropouts[3](out)

        ############ Classification Layer, by now, time is crunched down to 1
        out = out.squeeze() # get rid of singleton time
        out = self.classification(out)
        h, c = state

        #out = out.contiguous()
        #out = out.view(out.shape[0], -1)
        #out = self.classification(c[-1])

        return out


if __name__ == '__main__':
    print('testing rnn')

    test_X = Variable(torch.FloatTensor(
                        np.random.rand(123, 22, 1000)), requires_grad=True)
    test_Y = Variable(torch.LongTensor(
                        np.random.randint(4, size = (123))))

    rnn = Large_RNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adagrad(rnn.parameters(),
                                    weight_decay=0.01)

    out = rnn(test_X)
    print('forward works!')
    print(out.shape)
    loss = criterion(out, test_Y)
    loss.backward()
    print('backward works!')
