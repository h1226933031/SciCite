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
from our_code.models import LSTM, BiLSTM
torch.manual_seed(1)

class Model(nn.Module):
    def __init__(self, module_type='LSTM',
                 num_classes=3,
                 input_size=200,
                 hidden_size=[128, 64],
                 num_layers=1,
                 activat_func='relu',
                 dropout_rate=0,
                 batch_size = 128,
                 optimizer = 'adam',
                 valid_split = 0.1,
                 epoch = 100,
                 learning_rate = 0.01,
                 patient = 5):
        self.module_type = module_type
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #self.seq_length = seq_length
        self.activat_func = activat_func
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.valid_split = valid_split
        self.epoch = epoch
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.patient = patient

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # to store the train loss and validation loss
        self.train_loss = []
        self.valid_loss = []

        if self.module_type == 'LSTM':
            self.model = LSTM(num_classes=self.num_classes,
                              input_size=self.input_size,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              activat_func=self.activat_func,
                              dropout_rate=self.dropout_rate,
                              batch_size = self.batch_size).to(self.device)
        elif self.module_type == 'BiLSTM':
            self.model = BiLSTM(num_classes=self.num_classes,
                                input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                activat_func=self.activat_func,
                                dropout_rate=self.dropout_rate,
                                batch_size = self.batch_size).to(self.device)
        else:
            #TODO
            # This can be the RNN model
            pass

        if self.optimizer =="SGD":
            self.optimizer_fc = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        else:
            self.optimizer_fc = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.loss_fc = nn.CrossEntropyLoss()

    def load_data(self, X, y, valid_split=0.1):
        if valid_split == 0: #do not need to split the data
            data_train = [X]
            data_train.append(list(y))
            data_train = list(zip(*data_train))
            train_loader = DataLoader(data_train, batch_size=self.batch_size)
            return train_loader, None
        else:
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=valid_split)
            # training set
            X_train = [np.array(X_train)]
            X_train.append(list(y_train))
            X_train = list(zip(*X_train))
            train_loader = DataLoader(X_train, batch_size=self.batch_size)
            # validation set
            X_valid = [np.array(X_valid)]
            X_valid.append(list(y_valid))
            X_valid = list(zip(*X_valid))
            valid_loader = DataLoader(X_valid, batch_size=self.batch_size)
            return train_loader, valid_loader
    def train_model(self, X, y, path):
        min_val = np.inf
        train_loader, valid_loader = self.load_data(X, y=y, valid_split=self.valid_split)
        for i in range(self.epoch):
            total_train_loss = []
            count = 0
            self.model.train() # start the training process
            for step, (b_x, b_y) in enumerate(train_loader):
                b_x = b_x.type(torch.FloatTensor).to(self.device)
                # forward
                y_pred = self.model()
                loss = self.loss_fc(y_pred, b_y)
                # backward
                self.optimizer_fc.zero_grad()
                loss.backward()
                # update weight
                self.optimizer_fc.step()
                total_train_loss.appenda(loss.item())
            self.train_loss.append(np.mean(total_train_loss)) # the average cross entrophy
            print(f"Epoch {i}: Training Loss {np.mean(total_train_loss)}")

            # below is the validation part
            if self.valid_split != 0:
                total_val_loss = []
                self.model.eval() # start the training process
                for step, (b_x, b_y) in enumerate(valid_loader):
                    b_x = b_x.type(torch.FloatTensor).to(self.device)
                    with torch.no_grad():
                        # forward
                        y_pred = self.model(b_x)
                    loss = self.loss_fc(y_pred, b_y)
                    total_val_loss.appenda(loss.item())
                self.val_loss.append(np.mean(total_val_loss)) # the average cross entrophy
                print(f"Epoch {i}: Validation Loss {np.mean(total_val_loss)}")

                # if the new validation is smaller than
                if self.valid_loss[-1] < min_val:
                    torch.save({
                        'epoch': 5,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer_fc.state_dict()
                    }, path)
                    min_val = self.valid_loss[-1]
                else: count += 1

            # this is the early stopping step
            if count > self.patient:
                break

    def test_model(self, X, y, path):
        best_model = torch.load(path).get('model_state_dict').cuda()
        best_model.eval()
        X = torch.tensor(X, dtype=torch.long).cuda()
        y_pred = best_model(X)
        loss = self.loss_fc(y_pred, y)
        prediction = torch.argmax(y_pred, dim=1)
        correct = (prediction == y).sum().item()
        numb = len(X)  # 此处的8代表batch_size
        accuracy = (correct/numb)*100
        print(f"The accuracy on the testing dataset is {accuracy} and the loss is {loss.item()}")



