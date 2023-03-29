import torch
import torch.nn as nn
from RNN_BiLSTM.utils.data_loader import get_iters
import torch.optim as optim
import time
from RNN_BiLSTM.utils.utils import train, evaluate, epoch_time, adjust_learning_rate #, train_bert, evaluate_bert
from RNN_BiLSTM.models import CNN, BiLSTM_Attention, BERT, RNN
import matplotlib.pyplot as plt
#from scibert_model import data_preprocessing
#import argparse
import warnings
import numpy as np
#import json
#from scibert_model import model
warnings.filterwarnings("ignore")


class Model(nn.Module):

    def __init__(self, MODEL_NAME='RNN', BATCH_SIZE=256, PATH='rnn_batch_', lr=0.001, embedding_layer=100, train_data_path="./scicite-data/train.jsonl", valid_data_path="./scicite-data/dev.jsonl", test_data_path="./scicite-data/test.jsonl"):
        super(Model, self).__init__()
        self.BATCH_SIZE = 256
        self.EMBEDDING_DIM = embedding_layer
        self.MODEL_NAME = MODEL_NAME
        self.PATH = PATH
        # this part is to store the loss and accuracy while training the model
        self.total_train_loss = []
        self.total_valid_loss = []
        self.total_train_acc = []
        self.total_valid_acc = []
        self.lr = lr

        self.train_iter, self.val_iter, self.test_iter, self.TEXT, self.LABEL = get_iters(batch_size=BATCH_SIZE, train_data_path=train_data_path, test_data_path=test_data_path, valid_data_path=valid_data_path)
        self.INPUT_DIM = len(self.TEXT.vocab)
        # ----- choose the target model ----- #

        if self.MODEL_NAME == 'BERT':
            '''
            self.model = model.CustomBertClassifier(hidden_dim= 100, bert_dim_size=self.bert_dim_size, num_of_output=3, model_name=self.bertmodel_name)
            '''
        elif self.MODEL_NAME == 'CNN':
            # CNN hyper parameters
            N_FILTERS = 100
            FILTER_SIZES = [2, 3, 4]
            OUTPUT_DIM = len(self.LABEL.vocab)
            DROPOUT = 0.5
            # PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
            self.model = CNN.Model(self.INPUT_DIM, self.EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT,
                                   pad_idx=None)

        elif self.MODEL_NAME == 'Attn_BiLSTM':
            N_HIDDEN = 5  # number of hidden units in one cell
            NUM_CLASSES = 3
            self.model = BiLSTM_Attention.Model(self.INPUT_DIM, self.EMBEDDING_DIM, N_HIDDEN, NUM_CLASSES)

        elif self.MODEL_NAME == 'RNN':
            DROPOUT = 0.1
            N_HIDDEN = 32
            NUM_CLASSES = 3
            self.model = RNN.Model(dropout_rate=DROPOUT, vocab_size=self.INPUT_DIM, embedding_dim=self.EMBEDDING_DIM, hidden_size=N_HIDDEN, output_dim=NUM_CLASSES)

        else:
            print(f'model type {self.MODEL_NAME} is currently not supported.')
            exit()
        # ----- choose the target model ----- #

        # ----- because BERT is a END2END model, there is no need to load the pretrain embedding ----- #
        if MODEL_NAME not in ['BERT']:

            pretrained_embeddings = self.TEXT.vocab.vectors
            self.model.embedding.weight.data.copy_(pretrained_embeddings)

            # Then zero the initial weights of the unknown and padding tokens.
            UNK_IDX = self.TEXT.vocab.stoi[self.TEXT.unk_token]
            self.model.embedding.weight.data[UNK_IDX] = torch.zeros(self.EMBEDDING_DIM)
            # model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

        # ----- checking devices ----- #
        # if torch.cuda.is_available():
        #     print("Cuda is available, using CUDA")
        #     self.device = torch.device('cuda')
        # elif torch.backends.mps.is_available():
        #     print("MacOS acceleration is available, using MPS")
        #     self.device = torch.device('mps')
        # else:
        #     print("No acceleration device detected, using CPU")
        #     self.device = torch.device('cpu')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        if self.MODEL_NAME == "BERT":
            self.criterion = nn.NLLLoss()
        else:
            self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, patient=5, N_EPOCHS=100, early_stopping=True, save_best_model=True, lradj=None):
        patient_count = 0
        best_valid_loss = np.inf
        for epoch in range(N_EPOCHS):
            start_time = time.time()
            if self.MODEL_NAME == "BERT":
                '''
                train_loss, train_acc = train_bert(model=self.model, train_loader=self.train_iter, optimizer=self.optimizer, criterion=self.criterion, device=self.device, bz=self.BATCH_SIZE)
                valid_loss, valid_acc = evaluate_bert(model=self.model, data=self.val_iter, criterion=self.criterion)
                '''
            else:
                train_loss, train_acc = train(self.model, self.train_iter, self.optimizer, self.criterion, self.MODEL_NAME)
                valid_loss, valid_acc = evaluate(self.model, self.val_iter, self.criterion, self.MODEL_NAME)

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
                adjust_learning_rate(self.optimizer, epoch + 1, args)

    def test(self):  # this is to test the model on the testing dataset
        best_model = torch.load(f'./ckpt/{self.PATH}-model.pt').get('model_state_dict').cuda()
        test_loss, test_acc = evaluate(best_model, self.test_iter, self.criterion, self.MODEL_NAME)
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


# training
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
    model = Model(MODEL_NAME="RNN")
    model.train()

    model.test()
    model.plot()
