import math

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.nn import Parameter
from torch import Tensor
import torch.nn.functional as F

class gru_cell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(gru_cell, self).__init__()     # initializes parent class
        self.input_size = input_size    # defines input size of data
        self.hn_size = hidden_size      # defines hidden layer sizes
        self.bias = bias                # a boolean variable that states if the bias is added to the model
                                        # [TRUE = more flexibility, more parameters (overfitting), FALSE = reduces parameters but limits model ability to fit data
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)    # applies linear transormation to input data, size = 3*hidden_size since GRU has 3 gates
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)   # same as x2h with hidden layer as input
        self.reset_parameters()         # initiaizes weights

    def reset_parameters(self):                 # initializes weights / biases of x2h, h2h
                                                    # reason for attributing random weight value is to avoid symmetry == quicker, more accurate learning
        std = 1.0 / math.sqrt(self.hidden_size)     # gets standard deviation
        for w in self.paramters():
            w.data.uniform(-std, std)               # initializes every parameter w with rand val from uniform distribution

    def forward(self, x, hidden):       # defines the forward pass of the GRU, takes in input data 'x', previous input state 'hidden', and returns output 'hy'

        x = x.view(-1, x.size(1))       # reshapes data into 2D tensor

        gate_x = self.x2h(x)            # applies linear transformations to input 'x'
        gate_h = self.h2h(hidden)       # applies linear transformations to previous hidden layer 'hidden'

        gate_x = gate_x.squeeze()       # removes dimensions of size 1
        gate_h = gate_h.squeeze()       # removes dimensions of size 1

        i_r, i_i, i_n = gate_x.chunk(3, 1)          # i, r, n correspond to 'input', 'reset', 'new' gates
        h_r, h_i, h_n = gate_h.chunk(3, 1)          # splitting each x and hidden into each gate for processing

        resetgate = F.sigmoid(i_r + h_r)            # resetgate takes sum of input_reset and hidden_reset applied to sigmoid activation
        inputgate = F.sigmoid(i_i + h_i)            # inputgate takes sum of input_input and hidden_input applied to sigmpoid activation
        newgate = F.tanh(i_n + (resetgate * h_n))   # newgate takes sum of input_new and hidden_new x resetgate applied to tanh activation

        hy = newgate + inputgate * (hidden - newgate)   # finally output hy is sum and multiplication of gates with previous hidden state

class gru_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(gru_model, self).__init__()   # initializes parent class

        self.hidden_dim = hidden_dim        # dimensionality of hidden_layer
        self.layer_dim = layer_dim          # number of GRU layers stacked

        self.gru_cell = gru_cell(input_dim, hidden_dim, layer_dim)  # defining GRU cell with inputs: input dimension, hidden layer dimension, number of layers

        self.fc = nn.Linear(hidden_dim, output_dim)     # defining output layer with inputs: hidden dimensions, output dimensions

    def forward(self, x):       # forward pass for model
        if torch.cuda.isa():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())   # initializing hidden state with 0s using GPU
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))          # initializing hidden state without 0s

        outs = []       # outs is the output vector

        hn = h0[0,:,:]  # hn is the updated hidden state vector

        for seq in range(x.size(1)):
            hn = self.gru_cell(x[:,seq,:], hn)  # setting new hidden state
            outs.append(hn)                     # appending output with hidden sate

        out = outs[-1].squeeze()    # removing any dimensions equal to 1

        out = self.fc(out)          # getting final output vector

        return out