import torch
import numpy as np
import json
import pandas as pd
from utils.data_preprocessing import DataPreprocessing


def load_n_preprocess(json_path, processor, label_dic, csv_format=False):
    if csv_format:
        df = pd.read_csv(json_path)[['string', 'label']]
        texts = processor.preprocessing(list(df['string']))
        labels = np.array([label_dic[y] for y in list(df['label'])], dtype=np.int32)
    else:  # load json format
        jsons = [json.loads(x) for x in open(json_path, "r", encoding="utf-8")]  # a list of dict
        texts = processor.preprocessing([x['string'] for x in jsons])  # a list of lists
        try:
            labels = np.array([label_dic[x['label']] for x in jsons], dtype=np.int32)
        except:
            labels = [label_dic[x['label']] for x in jsons]
    return texts, labels


def build_embed_matrix(vector_path='./.vector_cache/glove.6B.50d.txt', n_dim=50, n_vocab=24000):
    # {0:'<UNK>', 1:'the', ...}
    ix2word, word2ix, vector_list = {0: "<UNK>"}, {"<UNK>": 0}, [np.zeros(n_dim)]
    with open(vector_path, mode='r', encoding='utf-8') as f:
        for j, line in enumerate(f.readlines()):
            line_list = line.split()
            word2ix[line_list[0]] = j + 1
            ix2word[j + 1] = line_list[0]
            wordvec = np.array([float(num) for num in line_list[1:]])
            vector_list.append(wordvec)
            if j > n_vocab:
                break
    vector_list = np.array(vector_list).reshape(-1, n_dim)
    return ix2word, word2ix, torch.FloatTensor(vector_list)


def encoding_tokens(train_data, word2ix, max_length):  # output: Tensor[n_sample, max_length]
    encoded_ids = []
    print("len(train_data)", len(train_data))
    if not max_length:  # batch_size = 1, used for visualization
        ids = [word2ix[token] if token in word2ix.keys() else 0 for token in train_data]
        encoded_ids = np.array(ids, dtype=int)
        return torch.IntTensor(encoded_ids).unsqueeze(0)

    for token_list in train_data:
        n = len(token_list)
        if n >= max_length:
            token_list = token_list[:max_length]
        else:
            token_list += ['<UNK>'] * (max_length - n)
        ids = [word2ix[token] if token in word2ix.keys() else 0 for token in token_list]
        encoded_ids.append(np.array(ids, dtype=int))
    encoded_ids = np.array(encoded_ids, dtype=int).reshape(-1, max_length)
    return torch.IntTensor(encoded_ids)


class Dataset(torch.utils.data.Dataset):

    def __init__(self, x, y):  # , mask, token_type_ids):
        self.x = x
        self.y = y
        # self.mask = mask
        # self.token_type_ids = token_type_ids

    def __getitem__(self, idx):
        # return (self.x[idx], self.mask[idx], self.token_type_ids[idx]), self.y[idx]
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


def get_iters(train_path, val_path, test_path, word2ix, embed_matrix, BATCH_SIZE=128, MAX_LENGTH=256):
    label_dic = {'background': 0, 'method': 1, 'result': 2}
    # load pre-trained models
    print("shape of embed_matrix:", embed_matrix.shape)  # embed_matrix: [n_vocab, n_dim]

    # load and preprocess raw texts
    dp = DataPreprocessing(contract=True, lemmatize=False, lowercase=True, stopword=False, stopword_set=None)
    # encode tokens into vector ids
    train_iter, val_iter, test_iter = None, None, None
    if train_path:
        train_texts, train_labels = load_n_preprocess(train_path, dp, label_dic, csv_format=True)
        train_inputs = encoding_tokens(train_data=train_texts, word2ix=word2ix, max_length=MAX_LENGTH)
        train_dataset = Dataset(train_inputs, torch.LongTensor(train_labels))
        train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    if val_path:
        val_texts, val_labels = load_n_preprocess(val_path, dp, label_dic)
        val_inputs = encoding_tokens(train_data=val_texts, word2ix=word2ix, max_length=MAX_LENGTH)
        val_dataset = Dataset(val_inputs, torch.LongTensor(val_labels))
        val_iter = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    if test_path:
        test_texts, test_labels = load_n_preprocess(test_path, dp, label_dic)
        test_inputs = encoding_tokens(train_data=test_texts, word2ix=word2ix, max_length=MAX_LENGTH)
        test_dataset = Dataset(test_inputs, torch.LongTensor(test_labels))
        test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # for batch in tqdm(train_iter):
    #     x, y = batch
    #     print(x.shape)  # [batch_size, seq_len]
    #     print(y.shape)  # [batch_size]
    #     break
    return train_iter, val_iter, test_iter
