import torch
import torch.nn as nn
from tqdm import tqdm
from torchmetrics import F1Score, Accuracy
from Bert.scibert_model.model import CustomBertClassifier
from Bert.scibert_model.data_preprocessing import bert_process
import json
import numpy as np
import torch.optim as optim
import time
from Bert.scibert_model.utils import evaluate_bert, train_bert, adjust_learning_rate, epoch_time, categorical_accuracy
import matplotlib.pyplot as plt
from collections import defaultdict, Counter, OrderedDict
import torch.nn.functional as F
import math

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class BertModel(nn.Module):

    def __init__(self, BATCH_SIZE=256, PATH='rnn_batch_', lr=0.001, embedding_layer=100, train_data_path="./scicite-data/train.jsonl", valid_data_path="./scicite-data/dev.jsonl", test_data_path="./scicite-data/test.jsonl"):
        super(BertModel, self).__init__()
        self.BATCH_SIZE = 256
        self.EMBEDDING_DIM = embedding_layer
        self.PATH = PATH
        # this part is to store the loss and accuracy while training the model
        self.total_train_loss = []
        self.total_valid_loss = []
        self.total_train_acc = []
        self.total_valid_acc = []
        self.lr = lr
        self.bertmodel_name = 'allenai/scibert_scivocab_uncased'
        if self.bertmodel_name == 'bert-base-uncased':
            self.bert_dim_size = 768
        elif self.bertmodel_name == 'allenai/scibert_scivocab_uncased':
            self.bert_dim_size = 768
        else:
            self.bert_dim_size = 1024
        def load_data(path):
            data = []
            for x in open(path, encoding='utf-8'):
                data.append(json.loads(x))
            return data
        train_data = load_data(train_data_path)
        dev_data = load_data(valid_data_path)
        test_data = load_data(test_data_path)
        #train_data, test_data, dev_data = load_data(train_data_path), load_data(test_data_path), load_data(valid_data_path)
        self.train_data = bert_process(train_data, batch_size=self.BATCH_SIZE, pretrained_model_name=self.bertmodel_name, confidence_level=0, cite2sentence_percent=1)
        # train = bert_process(train_data, train_data_sci ,batch_size=bz, pretrained_model_name=bertmodel_name, repeat=repeat)
        self.train_iter = self.train_data.data_loader
        print(len(self.train_data.data))

        self.dev_data = bert_process(dev_data, batch_size=self.BATCH_SIZE, pretrained_model_name=self.bertmodel_name, confidence_level=0, cite2sentence_percent=1)
        self.val_iter = self.dev_data.data_loader

        self.test_data = bert_process(test_data, batch_size=self.BATCH_SIZE, pretrained_model_name=self.bertmodel_name, confidence_level=0, cite2sentence_percent=1)
        self.test_iter = self.test_data.data_loader
        self.model = CustomBertClassifier(hidden_dim= 100, bert_dim_size=self.bert_dim_size, num_of_output=3, model_name=self.bertmodel_name)
        # ----- checking devices ----- #
        if torch.cuda.is_available():
            print("Cuda is available, using CUDA")
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            print("MacOS acceleration is available, using MPS")
            self.device = torch.device('mps')
        else:
            print("No acceleration device detected, using CPU")
            self.device = torch.device('cpu')

        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    def train(self, patient=5, N_EPOCHS=100, early_stopping=True, save_best_model=True, lradj=None):
        patient_count = 0
        best_valid_loss = np.inf
        for epoch in range(N_EPOCHS):
            start_time = time.time()
            print('start training')
            train_loss, train_acc = train_bert(model=self.model, train_loader=self.train_iter, optimizer=self.optimizer, criterion=self.criterion, device=self.device, bz=self.BATCH_SIZE)
            valid_loss, valid_acc = evaluate_bert(model=self.model, data=self.val_iter, criterion=self.criterion, data_object=self.dev_data, device=self.device)

            end_time = time.time()

            self.total_train_loss.append(train_loss)
            self.total_valid_loss.append(valid_loss)
            self.total_train_acc.append(train_acc)
            self.total_valid_acc.append(valid_acc)

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                patient_count = 0
                best_valid_loss = valid_loss
                if save_best_model:
                    torch.save(self.model.state_dict(), f'./ckpt/{self.PATH}-model.pt')
            else:
                patient_count += 1

            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
            # apply early stopping
            if patient_count > patient and early_stopping:
                break
            # apply learning rate decay
            if lradj:
                adjust_learning_rate(self.optimizer, epoch + 1, 'type1')
    def test(self):  # this is to test the model on the testing dataset
        best_model = torch.load(f'./ckpt/{self.PATH}-model.pt').get('model_state_dict').cuda()
        test_loss, test_acc = evaluate_bert(model=best_model, data=self.test_iter, criterion=self.criterion, device=self.device, data_object=self.test_data)
        print(f"The accuracy on the testing dataset is {test_acc} and the loss is {test_loss}")

    def plot(self):  # plot the model performance
        fig, ax = plt.subplots()
        ax.plot(range(len(self.total_train_loss)), self.total_train_loss, label='training loss')
        ax.plot(range(len(self.total_valid_loss)), self.total_valid_loss, label='validation loss')
        ax.set_xlabel('iteration')
        ax.set_ylabel('loss')
        ax.set_title('loss fig')
        ax.legend()
        plt.savefig(f'./fig/{self.PATH}-loss.png')

        fig, ax = plt.subplots()
        ax.plot(range(len(self.total_train_acc)), self.total_train_acc, label='training accuracy')
        ax.plot(range(len(self.total_valid_acc)), self.total_valid_acc, label='validation accuracy')
        ax.set_xlabel('iteration')
        ax.set_ylabel('accuracy')
        ax.set_title('accuracy fig')
        ax.legend()
        plt.savefig(f'./fig/{self.PATH}-accuracy.png')

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Experiments for Citation Text Multi-classification')
    #
    # # set experimental configs
    # parser.add_argument('--MODEL_NAME', type=str, default='Attn_BiLSTM',
    #                     help="model name, options: ['CNN', 'Attn_BiLSTM', 'RNN', 'BERT']")
    # parser.add_argument('--EMBEDDING_METHOD', type=str, default='glove', help="options: ['glove', 'word2vec']")
    # parser.add_argument('--EMBEDDING_DIM', type=int, default=100, help="the embed dimension you choose")
    # parser.add_argument('--OUTPUT_DIM', type=int, default=3, help='usually refers to num_classes')
    #
    # # set training configs
    # parser.add_argument('--N_EPOCHS', type=int, default=1)
    # parser.add_argument('--BATCH_SIZE', type=int, default=256)
    # parser.add_argument('--INITIAL_LR', type=float, default=0.0001)
    # parser.add_argument('--EARLY_STOPPING', type=bool, default=True)
    # parser.add_argument('--SAVE_BEST_MODEL', type=bool, default=True)
    # parser.add_argument('--lradj', type=str, default=None, help="options: [None, 'type1', 'type2', 'type3', 'type4']")
    #
    # args = parser.parse_args()
    # print('Args in experiment:')
    # print(args)
    #
    # model = Model(BATCH_SIZE=args.BATCH_SIZE, MODEL_NAME=args.MODEL_NAME, lr=args.INITIAL_LR,
    #               embedding_layer=args.EMBEDDING_DIM)
    # model.train(N_EPOCHS=args.N_EPOCHS, early_stopping=args.EARLY_STOPPING, save_best_model=args.SAVE_BEST_MODEL, lradj=args.lradj)
    model = BertModel()
    model.train()
    model.test()
    model.plot()


