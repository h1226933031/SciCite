import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
torch.manual_seed(1)

class LSTM(nn.Module):
    def __init__(self, num_classes=3, input_size=200, hidden_size=[128, 64], num_layers=1, activat_func='relu', dropout_rate=0, batch_size = 128):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #self.seq_length = seq_length
        self.activat_func = activat_func
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size

        # define the layer of the model
        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size[0], num_layers=num_layers, batch_first=True)
        self.drop = nn.Dropout(p=dropout_rate)
        self.fc_1 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc = nn.Linear(hidden_size[1], num_classes)
        self.hidden = self.init_hidden()

        if self.activat_func == 'relu':
            self.activate = nn.ReLU()
        elif self.activat_func == 'Tanh':
            self.activate = nn.Tanh()
        elif self.activat_func == 'sigmoid':
            sel.activate == nn.Sigmoid()
        else:
            self.activate = nn.ReLU()

    def init_hidden(self):
        return (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size[0])), # initial hidden state
                Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size[0])))
    def forward(self, X):
        # Forward Propagate input through LSTM
        output, self.hidden = self.LSTM(X, self.hidden)
        out = self.activate(output[:,-1,:])
        out = self.fc_1(out)
        out = self.activate(out)
        out = self.fc(out)
        # do the log prob
        #out = F.log_softmax(out)

        return out

class BiLSTM(nn.Module):
    def __init__(self, num_classes=3, input_size=200, hidden_size=[128, 64], num_layers=1, activat_func='relu', dropout_rate=0, batch_size = 128):
        super(BiLSTM, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #self.seq_length = seq_length
        self.activat_func = activat_func
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size

        # define the layer of the model
        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size[0], \
                            num_layers=num_layers, batch_first=True, bidirectional=self.bidirectional)
        self.drop = nn.Dropout(p=dropout_rate)
        self.fc_1 = nn.Linear(hidden_size[0]*2, hidden_size[1])
        self.fc = nn.Linear(hidden_size[1], num_classes)
        self.hidden = self.init_hidden()

        if self.activat_func == 'relu':
            self.activate = nn.ReLU()
        elif self.activat_func == 'Tanh':
            self.activate = nn.Tanh()
        elif self.activat_func == 'sigmoid':
            sel.activate == nn.Sigmoid()
        else:
            self.activate = nn.ReLU()

    def init_hidden(self):
        return (Variable(torch.zeros(self.num_layers*2, self.batch_size, self.hidden_size[0])), # initial hidden state
                Variable(torch.zeros(self.num_layers*2, self.batch_size, self.hidden_size[0])))
    def forward(self, X):
        # Forward Propagate input through LSTM
        output, self.hidden = self.LSTM(X, self.hidden)
        bi_concat = torch.cat([output[:,0,:-self.hidden_size[0]:], output[:,-1,:self.hidden_size[0]]], dim=1)
        out = self.activate(bi_concat)
        out = self.fc_1(out)
        out = self.activate(out)
        out = self.fc(out)
        # do the log prob
        #out = F.log_softmax(out)

        return out