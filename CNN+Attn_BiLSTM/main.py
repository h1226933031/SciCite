import torch
import torch.nn as nn
from utils.data_loader import get_iters
import torch.optim as optim
import time
from utils.utils import train, evaluate, epoch_time
from models import CNN, BiLSTM_Attention, BERT, RNN
import matplotlib.pyplot as plt

class Model():

    def __init__(self, MODEL_NAME='RNN', BATCH_SIZE=256, PATH='rnn_batch_'):
        self.BATCH_SIZE = 256
        self.train_iter, self.val_iter, self.test_iter, self.TEXT, self.LABEL = get_iters(batch_size=BATCH_SIZE)
        self.INPUT_DIM = len(self.TEXT.vocab)
        self.EMBEDDING_DIM = 100
        self.MODEL_NAME = MODEL_NAME
        self.PATH = PATH

        # this part is to store the loss and accuracy while training the model
        self.total_train_loss = []
        self.total_valid_loss = []
        self.total_train_acc = []
        self.total_valid_acc = []

        # ----- choose the target model ----- #
        if MODEL_NAME == 'BERT':
            self.model = BERT.Model()

        elif MODEL_NAME == 'CNN':
            # CNN hyper parameters
            N_FILTERS = 100
            FILTER_SIZES = [2, 3, 4]
            OUTPUT_DIM = len(self.LABEL.vocab)
            DROPOUT = 0.5
            # PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
            self.model = CNN.Model(self.INPUT_DIM, self.EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, pad_idx=None)

        elif MODEL_NAME == 'Attn_BiLSTM':
            N_HIDDEN = 5  # number of hidden units in one cell
            NUM_CLASSES = 3
            self.model = BiLSTM_Attention.Model(self.INPUT_DIM,self.EMBEDDING_DIM, N_HIDDEN, NUM_CLASSES)

        elif MODEL_NAME == 'RNN':
            DROPOUT = 0.1
            N_HIDDEN = 32
            NUM_CLASSES = 3
            self.model = RNN.Model(DROPOUT, self.INPUT_DIM, self.EMBEDDING_DIM, N_HIDDEN, NUM_CLASSES)

        else:
            print(f'model type {MODEL_NAME} is currently not supported.')
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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, patient=5, N_EPOCHS=100):
        patient_count = 0
        for epoch in range(N_EPOCHS):
            start_time = time.time()

            train_loss, train_acc = train(self.model, self.train_iter, self.optimizer, self.criterion, self.MODEL_NAME)
            valid_loss, valid_acc = evaluate(self.model, self.val_iter, self.criterion, self.MODEL_NAME)

            end_time = time.time()

            self.total_train_loss.append(train_loss)
            self.total_valid_loss.append(valid_loss)
            self.total_train_acc.append(train_acc)
            self.total_valid_acc.append(valid_acc)

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), f'./ckpt/{self.PATH}-model.pt')
            else:
                patient_count += 1

            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
            # apply early stopping
            if patient_count > patient:
                break
    def test(self): # this is to test the model on the testing dataset
        best_model = torch.load(f'./ckpt/{self.PATH}-model.pt').get('model_state_dict').cuda()
        test_loss, test_acc = evaluate(best_model, self.test_iter, self.criterion, self.MODEL_NAME)
        print(f"The accuracy on the testing dataset is {test_acc} and the loss is {test_loss}")

    def plot(self): # plot the model performance
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
    model = Model()
    model.train()
    model.test()
    model.plot()