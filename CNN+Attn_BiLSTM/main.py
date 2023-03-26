import torch
import torch.nn as nn
from utils.data_loader import get_iters
import torch.optim as optim
import time
from utils.utils import train, evaluate, epoch_time
from models import CNN, BiLSTM_Attention, BERT

# load batches first
BATCH_SIZE = 128
train_iter, val_iter, test_iter, TEXT, LABEL = get_iters(batch_size=BATCH_SIZE)
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100

# set model type
MODEL_NAME = 'Attn_BiLSTM'  # choose from ['CNN', 'Attn_BiLSTM', 'BERT']

# set model configs
if MODEL_NAME == 'BERT':
    pass
elif MODEL_NAME == 'CNN':
    # CNN hyper parameters
    N_FILTERS = 100
    FILTER_SIZES = [2, 3, 4]
    OUTPUT_DIM = len(LABEL.vocab)
    DROPOUT = 0.5
    # PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    model = CNN.Model(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, pad_idx=None)

elif MODEL_NAME == 'Attn_BiLSTM':
    N_HIDDEN = 5  # number of hidden units in one cell
    NUM_CLASSES = 3
    model = BiLSTM_Attention.Model(INPUT_DIM, EMBEDDING_DIM, N_HIDDEN, NUM_CLASSES)

else:
    print(f'model type {MODEL_NAME} is currently not supported.')
    exit()

# load pre-trained embeddings
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

# Then zero the initial weights of the unknown and padding tokens.
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
# model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

# set training configs
N_EPOCHS = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)


# training
if __name__ == '__main__':
    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train(model, train_iter, optimizer, criterion, MODEL_NAME)
        valid_loss, valid_acc = evaluate(model, val_iter, criterion, MODEL_NAME)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'./ckpt/{MODEL_NAME}-model.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

# from utils.data_preprocessing import DataPreprocessing, DataReader
# import numpy as np
#
# if __name__ == "__main__":
#     np.random.seed(112)
#
#     # Load and preprocess data
#     df = DataReader("./scicite-data/train.jsonl").read()
#     dp = DataPreprocessing(contract=True, lemmatize=False, lowercase=True, stopword=False, stopword_set=None)
#     df['string'] = dp.preprocessing(list(df['string']))