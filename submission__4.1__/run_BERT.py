import torch
import torch.nn as nn
from models.SciBERT import CustomBertClassifier
from utils.data_preprocessing import bert_process
import json
import numpy as np
import torch.optim as optim
import time
from utils.utils import evaluate_bert, train_bert, adjust_learning_rate, epoch_time, train_results_plot
import argparse

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class BertModel(nn.Module):

    def __init__(self, BATCH_SIZE=256, PATH='test', lr=0.001, embedding_layer=100,
                 train_data_path="./scicite-data/train.jsonl", valid_data_path="./scicite-data/dev.jsonl",
                 test_data_path="./scicite-data/test.jsonl"):
        super(BertModel, self).__init__()
        self.BATCH_SIZE = BATCH_SIZE
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

        self.train_data = bert_process(train_data, batch_size=self.BATCH_SIZE,
                                       pretrained_model_name=self.bertmodel_name, confidence_level=0,
                                       cite2sentence_percent=1)

        self.train_iter = self.train_data.data_loader
        print(len(self.train_data.data))

        self.dev_data = bert_process(dev_data, batch_size=self.BATCH_SIZE, pretrained_model_name=self.bertmodel_name,
                                     confidence_level=0, cite2sentence_percent=1)
        self.val_iter = self.dev_data.data_loader

        self.test_data = bert_process(test_data, batch_size=self.BATCH_SIZE, pretrained_model_name=self.bertmodel_name,
                                      confidence_level=0, cite2sentence_percent=1)
        self.test_iter = self.test_data.data_loader
        self.model = CustomBertClassifier(hidden_dim=100, bert_dim_size=self.bert_dim_size, num_of_output=3,
                                          model_name=self.bertmodel_name)
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

    def train(self, patient=5, N_EPOCHS=1, save_best_model=True, lradj=False):
        patient_count = 0
        best_valid_loss = np.inf
        for epoch in range(N_EPOCHS):
            start_time = time.time()
            print('start training')
            train_loss, train_acc = train_bert(model=self.model, train_loader=self.train_iter, optimizer=self.optimizer,
                                               criterion=self.criterion, device=self.device, bz=self.BATCH_SIZE)
            valid_loss, valid_acc = evaluate_bert(model=self.model, data=self.val_iter, criterion=self.criterion,
                                                  data_object=self.dev_data, device=self.device)

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
            if patient and patient_count > patient:
                break
            # apply learning rate decay
            if lradj:
                adjust_learning_rate(self.optimizer, epoch + 1, 'type1')

    def test(self):  # this is to test the model on the testing dataset
        best_model = CustomBertClassifier(hidden_dim=100, bert_dim_size=self.bert_dim_size, num_of_output=3,
                                          model_name=self.bertmodel_name)
        best_model.load_state_dict(torch.load(f'./ckpt/{self.PATH}-model.pt'))
        best_model.to(self.device)
        test_loss, test_acc = evaluate_bert(model=best_model, data=self.test_iter, criterion=self.criterion,
                                            device=self.device, data_object=self.test_data)
        print(f"The accuracy on the testing dataset is {test_acc} and the loss is {test_loss}")

    def plot(self):  # plot the model performance
        train_results_plot(self.total_train_loss, self.total_valid_loss,
                           self.total_train_acc, self.total_valid_acc, self.PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BERT Experiments for Citation Text Multi-classification')
    parser.add_argument('--TRAIN_PATH', type=str, default='./scicite-data/train.jsonl')
    parser.add_argument('--VAL_PATH', type=str, default='./scicite-data/dev.jsonl')
    parser.add_argument('--TEST_PATH', type=str, default='./scicite-data/test.jsonl')

    # # set experimental configs
    parser.add_argument('--EMBEDDING_LAYER', type=int, default=100, help="the embed layer you choose")

    # set training configs
    parser.add_argument('--N_EPOCHS', type=int, default=1)
    parser.add_argument('--BATCH_SIZE', type=int, default=256)
    parser.add_argument('--INITIAL_LR', type=float, default=0.001)
    parser.add_argument('--EARLY_STOPPING_PATIENT', type=int, default=5, help="If None then not apply early stopping")
    parser.add_argument('--SAVE_BEST_MODEL', type=bool, default=True)
    parser.add_argument('--lradj', type=str, default=None, help="options: [None, 'type1', 'type2', 'type3', 'type4']")

    args = parser.parse_args()
    print('Args in experiment:')
    print(args)

    model = BertModel(args.BATCH_SIZE, args.INITIAL_LR, args.EMBEDDING_LAYER, args.TRAIN_PATH, args.VAL_PATH,
                      args.TEST_PATH)
    model.train(args.EARLY_STOPPING_PATIENT, args.N_EPOCHS, args.SAVE_BEST_MODEL, args.lradj)
    model.plot()
    model.test()
