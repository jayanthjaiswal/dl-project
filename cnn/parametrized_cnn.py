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

class Large_CNN(nn.Module):

    def __init__(self,
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
        use_dropout             = False,
        verbose                 = False):
        '''
        Our implementation of:
            the large network
            from:
                Deep learning with convolutional neural networks
                for brain mapping and decoding of
                movement-related information from the human
                EEG
                -- shortname: Convolutional neural networks in EEG analysis
            by:
                Robin Tibor Schirrmeister,
                Jost Tobias Springenberg,
                Lukas Dominique Josef Fiederer,
                Martin Glasstetter,
                Katharina Eggensperger,
                Michael Tangermann,
                Frank Hutter
                Wolfram Burgard,
                Tonio Bal
            August 2017
            [link: https://arxiv.org/pdf/1703.05051.pdf]
        '''

        super(Large_CNN, self).__init__()

        self.batchnorm2_channels = batchnorm2_channels
        self.use_dropout = use_dropout

        # Size computations

        # Network

        self.norm_layer = nn.BatchNorm1d(eeg_channels)

        ############ Block 1

        # T here should be 1000 for our dataset (input)
        ## Convolutional Pass 1
        if (not kernels_get_all_channels):
            self.conv1 = nn.Conv2d( in_channels     = 1,
                                    out_channels    = conv1_out_channels,
                                    kernel_size     = (1,
                                                        conv1_kernel_size))

            self.conv2 = nn.Conv2d( in_channels     = conv1_out_channels,
                                    out_channels    = conv2_out_channels,
                                    kernel_size     = (eeg_channels, 1))
        else:
            self.conv1 = nn.Conv2d( in_channels     = 1,
                                    out_channels    = (eeg_channels
                                                       *conv1_out_channels),
                                    # to match the channels*activations
                                    # size we make more out channels
                                    kernel_size     = (eeg_channels,
                                                        conv1_kernel_size))

            self.conv2 = nn.Conv2d( in_channels    = (eeg_channels
                                                       *conv1_out_channels),
                                    out_channels    = conv2_out_channels,
                                    kernel_size     = (eeg_channels
                                                       *conv1_out_channels))

        conv1_out_timesteps = input_timesteps - conv1_kernel_size + 1

        if (self.batchnorm2_channels):
            self.norm2_opt1 = nn.BatchNorm1d(conv2_out_channels)
        else:
            self.norm2_opt2 = nn.BatchNorm1d(conv1_out_timesteps)

        self.elu2 = nn.ELU()

        self.maxpool2 = nn.MaxPool1d( kernel_size   = pool2_kernel_size,
                                      stride        = pool2_stride)

        ############ Block 2

        self.block2 = nn.Sequential(
                            nn.Conv1d( in_channels     = conv2_out_channels,
                                       out_channels    = conv3_out_channels,
                                       kernel_size     = conv3_kernel_size),
                            nn.BatchNorm1d(conv3_out_channels),
                            nn.ELU(),
                            nn.MaxPool1d( kernel_size   = pool3_kernel_size,
                                          stride        = pool3_stride))

        ############ Block 3

        self.block3 = nn.Sequential(
                            nn.Conv1d( in_channels  = conv3_out_channels,
                                       out_channels = conv4_out_channels,
                                       kernel_size  = conv4_kernel_size),
                            nn.BatchNorm1d(conv4_out_channels),
                            nn.ELU(),
                            nn.MaxPool1d( kernel_size   = pool4_kernel_size,
                                          stride        = pool4_stride))

        ############ Block 4

        self.block4 = nn.Sequential(
                            nn.Conv1d( in_channels  = conv4_out_channels,
                                       out_channels = conv5_out_channels,
                                       kernel_size  = conv5_kernel_size),
                            nn.BatchNorm1d(conv5_out_channels),
                            nn.ELU(),
                            nn.MaxPool1d( kernel_size   = pool5_kernel_size,
                                          stride        = pool5_stride))

        # Classification Layer, by now, time is crunched down to 1
        self.classification = nn.Sequential(
                                nn.Linear(conv5_out_channels, num_classes),
                                nn.Softmax(dim=0))
        
        if (use_drop):
            self.dropouts = [nn.Dropout() for i in range(4)]
   
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

        ############ Block 2
        out = self.block2(out)

        if (self.use_dropout):
            out = self.dropouts[1](out)

        ############ Block 3
        out = self.block3(out)

        if (self.use_dropout):
            out = self.dropouts[2](out)

        ############ Block 4
        out = self.block4(out)

        if (self.use_dropout):
            out = self.dropouts[3](out)

        ############ Classification Layer, by now, time is crunched down to 1
        out = out.squeeze() # get rid of singleton time
        out = self.classification(out)

        return out

def Train(X_data, Y_data):
    cnn = Large_CNN()

    num_epochs = 20
    batch_size = 10
    learning_rate = 3e-4
    weight_decay = 0.03

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
                    cnn.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay)
    
    train_acc = []
    val_acc = []
    loss_history = []
              
    gc.collect()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            gc.collect()
            images = Variable(images, requires_grad=True) #unsqueeze used to make a 4d tensor because 
            labels = Variable(labels, volatile=True)
    
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn(images)
            loss = criterion(outputs, labels)
            loss_history.append(loss)
            loss.backward()
            optimizer.step()
            del outputs
            del images, labels
            gc.collect()
    
            if (i+1) % 20 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                       %(epoch+1, num_epochs, i+1, len(cnn_training_data_X.shape)//batch_size, loss.data[0]))
                gc.collect()
                max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                print("{:.2f} MB".format(max_mem_used /(1024*1024)))
    
            del loss
        
        gc.collect()
        cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
        
        def test(X_data, Y_data, name='Train'):
            test_images = Variable(torch.Tensor(X_data), volatile=True)
            test_labels = torch.LongTensor(Y_data)
            
            test_outputs = cnn(test_images)
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_total = test_labels.size(0)
            test_correct = (test_predicted == test_labels).double().sum()
            test_acc = 100.0 * float(test_correct) / float(test_total)
            print('label set: %s predicted set: %s ----- %s Accuracy: %s %%' % (
                   np.unique(test_labels.data), np.unique(test_predicted.data), name, test_acc))
            
            val_acc.append(test_acc)
            del test_outputs
            del test_images, test_labels
            gc.collect()
        
        test(cnn_training_data_X, cnn_training_data_Y, 'Train')
        test(cnn_validation_data_X, cnn_validation_data_Y, 'Val')

if __name__ == '__main__':
    print('testing cnn')

    test_X = Variable(torch.FloatTensor(
                        np.random.rand(123, 22, 1000)), requires_grad=True)
    test_Y = Variable(torch.LongTensor(
                        np.random.randint(4, size = (123))))

    cnn = Large_CNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adagrad(cnn.parameters(),
                                    weight_decay=0.01)

    out = cnn(test_X)
    print('forward works!')
    loss = criterion(out, test_Y)
    loss.backward()
    print('backward works!')
