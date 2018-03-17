import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
from collections import OrderedDict

class CNN(nn.Module):

    def __init__(self,
        eeg_channels            = 22,
        eeg_repeats             = 2,
        input_timesteps         = 1000,
        conv1_kernel_size       = 25,
        pool1_kernel_size       = 75,
        pool1_stride            = 15,
        conv2_out_channels      = 30,
        conv2_kernel_size       = 3,
        outut_fc_layer_sizes    = [100],
        num_classes             = 4,
        verbose                 = False):
        '''
        '''

        super(CNN, self).__init__()

        # Size computations
        conv1_out_channels = eeg_channels * eeg_repeats
        # no padding, no stride, do dilation, so timesteps out is easy
        conv1_out_timesteps = input_timesteps - (conv1_kernel_size - 1)

        avgpool_out_timesteps = np.floor(
                                    (conv1_out_timesteps - pool1_kernel_size
                                    ) / (1.0 * pool1_stride) + 1)

        conv2_out_timesteps = avgpool_out_timesteps - (conv2_kernel_size - 1)

        flattened_size = conv2_out_timesteps * conv2_out_channels

        self.layer1 = nn.Sequential(
            ## Initial Nomalization Pass (for dataset)
            nn.BatchNorm1d(eeg_channels),

            # T here should be 1000 for our dataset (input)
            ## Convolutional Pass 1
            nn.Conv1d(  in_channels     = eeg_channels,
                        out_channels    = conv1_out_channels,
                        kernel_size     = conv1_kernel_size,
                        groups          = eeg_channels),
            # T here should be 976 for our dataset (after conv1)

            ## Nomalization, Nonlinearity, and Pooling Pass 1
            nn.BatchNorm1d(conv1_out_channels), 
            nn.ELU(),
            nn.AvgPool1d(kernel_size    = pool1_kernel_size,
                         stride         = pool1_stride),
            # T here should be 61 for our dataset (after avg)

            ## Convolutional Pass 2
            nn.Conv1d(  in_channels     = conv1_out_channels,
                        out_channels    = conv2_out_channels,
                        kernel_size     = conv2_kernel_size),
            # T here should be 59 for our dataset (after conv2)

            ## Nomalization, Nonlinearity
            nn.BatchNorm1d(conv2_out_channels),
            nn.ELU())

        # flattened_size here should be 1770 = 59*30 for our defaults
        prev_size = flattened_size

        fc_layers = OrderedDict()
        for idx, size in enumerate(outut_fc_layer_sizes):

            fc_layers['fc_{}'.format(4+idx)] = nn.Sequential(
                nn.Linear(prev_size, size),
                nn.BatchNorm1d(size),
                nn.ReLU())

            prev_size = size

        fc_layers['fc_final'] = nn.Linear(prev_size, num_classes)
        self.fcs = nn.Sequential(fc_layers)

        self.output = nn.Softmax(dim=0)
        
    #basic forward - go through two conv layers + fc layer
   
    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.size(0), -1)
        out = self.fcs(out)
        out = self.output(out)
        return out

if __name__ == '__main__':
    print('testing cnn')

    test_X = Variable(torch.FloatTensor(
                        np.random.rand(123, 22, 1000)), requires_grad=True)
    test_Y = Variable(torch.LongTensor(
                        np.random.randint(4, size = (123))))

    cnn = CNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adagrad(cnn.parameters(),
                                    weight_decay=0.01)

    out = cnn(test_X)
    print('forward works!')
    loss = criterion(out, test_Y)
    loss.backward()
    print('backward works!')
