from models import BiLSTM_Attention, RNN
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
import time
from utils.utils import train, evaluate, epoch_time, adjust_learning_rate, train_results_plot
from utils.data_loader import build_embed_matrix, get_iters
import argparse
import warnings
warnings.filterwarnings("ignore")


def main(args):
    # get embed matrix and load data
    _, word2ix, embed_matrix = build_embed_matrix(vector_path=args.EMBED_PATH, n_dim=args.EMBED_DIM, n_vocab=24000)

    train_iter, val_iter, test_iter = get_iters(args.TRAIN_PATH, args.VAL_PATH, args.TEST_PATH, word2ix,
                                                embed_matrix, args.BATCH_SIZE, args.MAX_LENGTH)

    # # ----- checking devices ----- #
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # training configs
    if args.MODEL_NAME == "Attn_BiLSTM":
        model = BiLSTM_Attention.Model(embed_matrix=embed_matrix, n_hidden=args.N_HIDDEN, num_classes=args.OUTPUT_DIM)
    elif args.MODEL_NAME == "RNN":
        model = RNN.Model(embed_matrix=embed_matrix, dropout_rate=args.DROP_OUT, n_hidden=args.N_HIDDEN,
                          num_classes=args.OUTPUT_DIM)
    else:
        print(f'model type {args.MODEL_NAME} is currently not supported.')
        exit()

    optimizer = optim.Adam(model.parameters(), lr=args.INITIAL_LR)
    criterion = nn.CrossEntropyLoss()

    total_train_loss, total_valid_loss, total_train_acc, total_valid_acc = [], [], [], []

    patient_count = 0
    best_valid_loss = np.inf

    for epoch in range(args.N_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train(model, train_iter, optimizer, criterion, model_name=args.MODEL_NAME)
        valid_loss, valid_acc = evaluate(model, val_iter, criterion, model_name=args.MODEL_NAME)

        end_time = time.time()

        total_train_loss.append(train_loss)
        total_valid_loss.append(valid_loss)
        total_train_acc.append(train_acc)
        total_valid_acc.append(valid_acc)

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            patient_count = 0
            best_valid_loss = valid_loss
            if args.SAVE_BEST_MODEL:
                torch.save(model, f'./ckpt/{args.MODEL_NAME}-best_model.pth')
        else:
            patient_count += 1

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

        # apply early stopping
        if args.EARLY_STOPPING_PATIENT and patient_count > args.EARLY_STOPPING_PATIENT:
            break

        # # apply learning rate decay:
        if args.lradj:
            adjust_learning_rate(optimizer, epoch + 1, args)

    # evaluate the best model
    best_model = torch.load(f'./ckpt/{args.MODEL_NAME}-best_model.pth')
    test_loss, test_acc = evaluate(best_model, test_iter, criterion, model_name=args.MODEL_NAME)
    print(f"The accuracy on the testing dataset is {test_acc * 100:.2f}% and the loss is {test_loss:.3f}.")
    # plot training performance
    train_results_plot(args.MODEL_NAME, total_train_loss, total_valid_loss,
                       total_train_acc, total_valid_acc, save_path=f'./fig/{args.MODEL_NAME}-')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Attn_BiLSTM & RNN Experiments for Citation Text Multi-classification')

    # set data_loader params
    parser.add_argument('--TRAIN_PATH', type=str, default='./scicite-data/train_balanced.csv')
    parser.add_argument('--VAL_PATH', type=str, default='./scicite-data/dev.jsonl')
    parser.add_argument('--TEST_PATH', type=str, default='./scicite-data/test.jsonl')
    parser.add_argument('--EMBED_PATH', type=str, default='./.vector_cache/glove.6B.50d.txt', help="vector file path")
    parser.add_argument('--EMBED_DIM', type=int, default=50, help="the embed dimension you choose")
    parser.add_argument('--OUTPUT_DIM', type=int, default=3, help='usually refers to num_classes')
    parser.add_argument('--MAX_LENGTH', type=int, default=256, help="cut texts into a fixed length")

    # set experimental configs
    parser.add_argument('--MODEL_NAME', type=str, default='Attn_BiLSTM',
                        help="model name, options: ['Attn_BiLSTM', 'RNN']")
    parser.add_argument('--N_HIDDEN', type=int, default=10, help='dimension of hidden states for BiLSTM & RNN')
    parser.add_argument('--DROP_OUT', type=float, default=0.2, help='dropout rate only for RNN')

    # set training configs
    parser.add_argument('--N_EPOCHS', type=int, default=1)
    parser.add_argument('--BATCH_SIZE', type=int, default=128)
    parser.add_argument('--INITIAL_LR', type=float, default=0.005)
    parser.add_argument('--EARLY_STOPPING_PATIENT', type=int, default=None, help="If None then not apply early stopping")
    parser.add_argument('--SAVE_BEST_MODEL', type=bool, default=True)
    parser.add_argument('--lradj', type=str, default=None, help="options: [None, 'type1', 'type2', 'type3', 'type4']")
    #
    args = parser.parse_args()
    print('Args in experiment:')
    print(args)

    main(args)
