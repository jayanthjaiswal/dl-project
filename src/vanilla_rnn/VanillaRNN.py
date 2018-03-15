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
            final_hidden_layer_sizes = [], # not including num_classes output layer
            num_classes = _default_num_classes,
            verbose = False):

        super(VanillaRNN, self).__init__()

        self.num_classes = num_classes
        self.conv_layers = conv_layers
        self.recurrent_hidden_size = recurrent_hidden_size
        self.recurrent_layer_num = recurrent_layer_num
        self.verbose = verbose

        prev_size = num_electrodes

        # convolutional processing layers -- optional --
        
        if (self.conv_layers):
            self.layer1 = nn.Sequential(
                nn.Conv1d(22, 32, kernel_size=5, padding=2, stride=1), #32x1000
                nn.MaxPool1d(2), #32 x 500
                nn.Conv1d(32, 32, kernel_size=4, padding=1, stride=2), #8x250
                nn.MaxPool1d(2)) #32 x 125
                #nn.Conv1d(32, 16, kernel_size=4, padding=2, stride=2), #8x250
                #nn.MaxPool1d(2)) #32 x 61
                #nn.BatchNorm1d(32), 
                #nn.ReLU(),
            #self.layer2 = nn.Sequential(
            #    nn.Conv1d(32, 8, kernel_size=4, padding=1, stride=2), #8x250
            #    nn.BatchNorm1d(8),
            #    nn.ReLU())

            prev_size = 32

        self.T = 125

        # input processing layers -- optional --

        # first layers converts the eeg channels to feature vectors for
        # the rnn input
        # these layers are all fully connected
        initial_layers = OrderedDict() 
        for idx, s in enumerate(initial_hidden_layer_sizes):
            initial_layers['initial_lin_{}'.format(idx)] = nn.Linear(prev_size, s)
            initial_layers['initial_relu_{}'.format(idx)] = nn.ReLU()
            prev_size = s

        #print(prev_size)
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
        output_layers['output'] = nn.Linear(prev_size, num_classes)

        self.output_layer = nn.Sequential(output_layers)

    def initialize_weights(self):
        for (k, weight) in self.rnn_layer.named_parameters():
            if 'weight_hh' in k:
                t = weight.data
                layer = Variable(nn.init.eye(t), requires_grad=True)

    def loss_regularizer(self):
        #grads = []
        #for (k, weight) in self.rnn_layer.named_parameters():
        #    if 'weight_hh' in k:
        #        if (weight.grad is not None):
        #            grads.append(weight.grad)

        #grads.reverse()

        loss = 0

        dLdh = self.rnn_out.grad.sum(dim=0).chunk(3)[-1]

        print(dLdh.shape)

        l2norm = nn.MSELoss()

        w = dict(self.rnn_layer.named_parameters())['weight_hh_l{}'.format(
                                                self.recurrent_layer_num - 1)]
        w = w.permute(1,0)

        print(w.shape)

        target = dtype(np.zeros_like(dLdh))
        prev_norm = l2norm(dLdh, target)**0.5
        for i in range(self.T):
            #print('dLdh[{}]: {}'.format(i, dLdh.sum()))
            dLdh = w.mv(dLdh)
            temp_norm = l2norm(dLdh, target)**0.5
            #print('prev norm[{}]: {}'.format(i, prev_norm))
            #print('temp norm[{}]: {}'.format(i, temp_norm))
            loss += (temp_norm/prev_norm -1)**2
            #dLdh_l.append(params[i])

        return loss

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
            #out = self.layer2(out)

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

        #initial_state = dtype(np.ones((1, out.shape[1],
        #                                self.recurrent_hidden_size))
        #                                /float(self.recurrent_hidden_size))

        #out, h = self.rnn_layer(out, initial_state) # now has shape T, N, H_rnn
        out, h = self.rnn_layer(out) # now has shape T, N, H_rnn

        if (self.recurrent_layer_num > 1):
            print('multilayer_h_shape {}'.format(h.shape))
            h = h.permute(1, 0, 2).contiguous()
            print('permuted_h_shape {}'.format(h.shape))
            h = h.view(h.shape[0], -1)
            print('flattened_h_shape {}'.format(h.shape))
        # H_rnn is the size of the hidden output

        if (self.verbose):
            print('outshape before extract {}'.format(out.shape))
            print('hidden state shape before extract {}'.format(h.shape))
            print('diff between last out and h: {}'.format(torch.mean(out.data[-1,:,:] - h.data)))
            print('sanity check h: {}'.format(torch.mean(h.data)))

        out = h.squeeze()

        out.retain_grad()
        self.rnn_out = out

        if (self.verbose):
            print('before final outshape {}'.format(out.shape))
        out = self.output_layer(out) # now has shape T, N, num_classes
        if (self.verbose):
            print('final outshape {}'.format(out.shape))

        return out, h
