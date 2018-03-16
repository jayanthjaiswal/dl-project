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

#from torch.nn.utils.rnn import pack_padded_sequence as ppseq
from collections import OrderedDict

class VanillaRNN(nn.Module):

    _default_num_classes = 4

    #def

    def sequence_loss(loss_fn, outputs, labels):
        '''
        returns a loss, which prioritizes the latest outputs
        '''
        pass
        #weights = np.array
        #loss = loss_fn(outputs[0,:], labels[0])
        #for i in range(1, leb(labels)):
        #    loss += 


    def __init__(self,
            num_electrodes = 22,
            conv_layers = False,
            initial_hidden_layer_sizes = [],
            recurrent_hidden_size = _default_num_classes,
            recurrent_layer_num = 1,
                # generally recommended 2/3, but in class we used 1
            recurrent_use_bias = False,
            recurrent_dropout = False,
            use_initial_state = False,
            final_hidden_layer_sizes = [], # not including num_classes output layer
            num_classes = _default_num_classes,
            verbose = False):

        super(VanillaRNN, self).__init__()

        self.num_classes = num_classes
        self.conv_layers = conv_layers
        self.recurrent_hidden_size = recurrent_hidden_size
        self.recurrent_layer_num = recurrent_layer_num
        self.use_initial_state = use_initial_state
        self.verbose = verbose

        prev_size = num_electrodes

        # convolutional processing layers -- optional --
        
        if (self.conv_layers):
            self.layer1 = nn.Sequential(
                nn.BatchNorm1d(22),
                nn.Conv1d(22, 16*22, kernel_size=25, padding=5, stride=5,
                            groups = 22), #32x1000
                nn.Conv1d(22, 16*22, kernel_size=25, padding=1, stride=5,
                            groups = 22))#, #32x1000
                #nn.Conv1d(64, 16, kernel_size=1, padding=1, stride=1), #32x1000
                #nn.BatchNorm1d(8))

            self.layer2 = nn.Conv1d(36, 36, kernel_size=

            #prev_size = 16
            #self.T = 32
            prev_size = 8
            self.T = 36

        # input processing layers -- optional --

        # first layers converts the eeg channels to feature vectors for
        # the rnn input
        # these layers are all fully connected
        initial_layers = OrderedDict() 
        for idx, s in enumerate(initial_hidden_layer_sizes):
            initial_layers['initial_lin_{}'.format(idx)] = nn.Linear(prev_size, s)
            initial_layers['initial_relu_{}'.format(idx)] = nn.ReLU()
            prev_size = s

        #initial_layers['batchnorm_before_rnn'] = nn.BatchNorm1d(prev_size)

        self.initial_layer = nn.Sequential(initial_layers)

        # the recurrent layers
        self.rnn_layer = nn.RNN(
                            input_size = prev_size, 
                            hidden_size = recurrent_hidden_size,
                            num_layers = recurrent_layer_num,
                            nonlinearity = 'tanh',
                            bias = recurrent_use_bias,
                            batch_first = False,
                            dropout = recurrent_dropout)

        prev_size = recurrent_hidden_size * recurrent_layer_num

        # output processing layers -- optional --
        output_layers = OrderedDict() 
        for idx, s in enumerate(final_hidden_layer_sizes):
            output_layers['final_lin_{}'.format(idx)] = nn.Linear(prev_size, s)
            output_layers['final_relu_{}'.format(idx)] = nn.ReLU()
            prev_size = s

        # output classifier layers
        output_layers['preoutput_ batchnorm'] = nn.BatchNorm1d(prev_size)
        output_layers['output'] = nn.Linear(prev_size, num_classes)

        self.output_layer = nn.Sequential(output_layers)

    def initialize_weights(self):
        for (k, weight) in self.rnn_layer.named_parameters():
            if 'weight_hh' in k:
                t = weight.data
                layer = Variable(nn.init.eye(t), requires_grad=True)

    def loss_regularizer(self):
        loss1 = 0

        dLdh = self.rnn_out.grad.sum(dim=0).chunk(self.recurrent_layer_num)[-1]

        l2norm = nn.MSELoss(size_average=False)

        w = dict(self.rnn_layer.named_parameters())['weight_hh_l{}'.format(
                                                self.recurrent_layer_num - 1)]
        w = w.permute(1,0)

        target = dtype(np.zeros_like(dLdh))
        prev_norm = l2norm(dLdh, target)**0.5
        for i in range(self.T):
            dLdh = w.mv(dLdh)
            temp_norm = l2norm(dLdh, target)**0.5
            loss1 += (temp_norm/prev_norm -1)**2

        #####  general weight magnitude

        loss2 = 0
        for (k, weight) in self.named_parameters():
            if 'weight' in k:
                target = dtype(np.zeros(weight.shape))
                loss2 += l2norm(weight, target)**0.5

        return loss1, loss2

    def forward(self, X):
        '''
        X should have shape (num samples, num_sensors=25, seq_len=1000)
        '''
        N, E, T = X.shape

        out = X

        if (not self.training):
            plt.plot(out.data[0, 0, :])
            plt.show()

        if (self.conv_layers):
            out = self.layer1(out)

        if (not self.training):
            plt.plot(out.data[0, 0, :])
            plt.show()

        # for rnn, we need to permute the input
        #   before it was N, E, T
        #   it needs to be T, N, E for rnn
        if (self.verbose):
            print('outshape1 {}'.format(out.shape))
        out = out.permute(2,0,1) # has shape T, N, E
        if (self.verbose):
            print('outshape2 {}'.format(out.shape))

        out = self.initial_layer(out) # has shape T, N, H_in
        # H_in is the last hidden network layer size

        if (self.use_initial_state):
            initial_state = dtype(np.ones((self.recurrent_layer_num,
                                            out.shape[1],
                                            self.recurrent_hidden_size))
                                            /float(self.recurrent_hidden_size))

            out, h = self.rnn_layer(out, initial_state) # now has shape T, N, H_rnn
        else:
            out, h = self.rnn_layer(out) # now has shape T, N, H_rnn

        if (self.recurrent_layer_num > 1):
            h = h.permute(1, 0, 2).contiguous()
            h = h.view(h.shape[0], -1)
        # H_rnn is the size of the hidden output

        if (self.verbose):
            print('outshape before extract {}'.format(out.shape))
            print('hidden state shape before extract {}'.format(h.shape))

        out = h.squeeze()

        out.retain_grad()
        self.rnn_out = out

        if (self.verbose):
            print('before final outshape {}'.format(out.shape))

        out = self.output_layer(out) # now has shape T, N, num_classes

        if (self.verbose):
            print('final outshape {}'.format(out.shape))

        return out, h
