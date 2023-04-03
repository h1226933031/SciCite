import torch
import argparse
import matplotlib.pyplot as plt
from utils.data_loader import load_n_preprocess, build_embed_matrix, get_iters, encoding_tokens, Dataset
from utils.data_preprocessing import DataPreprocessing
import seaborn as sns
import numpy as np


# def main(args):
#     get_iters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Attn_BiLSTM & RNN Experiments for Citation Text Multi-classification')
    parser.add_argument('--MODEL_NAME', type=str, default='Attn_BiLSTM')
    parser.add_argument('--EMBED_DIM', type=int, default=50)
    parser.add_argument('--EMBED_PATH', type=str, default='./.vector_cache/glove.6B.50d.txt', help="vector file path")
    parser.add_argument('--TEST_PATH', type=str, default='./scicite-data/dev.jsonl')
    args = parser.parse_args()

    # get embed matrix and load data
    label_dic = {'background': 'background', 'method': 'method', 'result': 'result'}
    dp = DataPreprocessing(contract=True, lemmatize=False, lowercase=True, stopword=False, stopword_set=None)
    _, word2ix, embed_matrix = build_embed_matrix(vector_path=args.EMBED_PATH, n_dim=args.EMBED_DIM, n_vocab=24000)

    # _, _, test_iter = get_iters(train_path=None, val_path=None, test_path=args.TEST_PATH, word2ix=word2ix,
    #                             embed_matrix=embed_matrix, BATCH_SIZE=1, MAX_LENGTH=None)

    # texts, labels = load_n_preprocess(args.TEST_PATH, dp, label_dic, csv_format=False)

    best_model = torch.load(f'./ckpt/{args.MODEL_NAME}-best_model.pth')
    best_model.eval()

    test_texts, test_labels = load_n_preprocess(args.TEST_PATH, dp, label_dic, csv_format=False)
    predictions, citations, labels = [], [], []

    with torch.no_grad():
        for text, label in zip(test_texts, test_labels):
            if 10 > len(text) >= 5:
                ids = encoding_tokens(train_data=text, word2ix=word2ix, max_length=None)
                predictions.append(best_model(ids)[1].numpy())
                print("weights shape:", len(predictions[-1]))
                citations.append(text)
                labels.append(label)
                if len(predictions) >= 3: break

    # visualization
    fig, axes = plt.subplots(ncols=1, nrows=3, figsize=(20, 30))
    for i, ax in enumerate(axes.flat):
        ax.matshow(predictions[i], cmap=plt.get_cmap('Greens'), alpha=0.8)
        ax.set_xticks(np.arange(len(citations[i])), citations[i])
        ax.set_yticks([])
        # ax.set_title(f'Attention weights for sample in class: {labels[i]}')
    # plt.legend()
    plt.savefig('attention weights visualization.png')
    plt.show()