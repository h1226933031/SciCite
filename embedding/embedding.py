import gensim
from gensim.models import word2vec
import pickle
import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids


class SciCiteEmbedding:
    def __init__(self, glove=False, word2vec=False, elmo=False, bert=False):
        self.word2vec = word2vec
        self.glove = glove
        self.elmo = elmo
        self.bert = bert
        self.ndim = None

    def embed(self, X_train):  # return a 3D numpy array. shape: [samples, max_length, embed_dim]
        embed_list = []
        if self.word2vec:  # time-consuming and could miss some tokens that do not appear in word2vec keys.
            # if word2vec model has not been dowloaded, plz run this:
            # path = gensim.downloader.load("word2vec-google-news-300", return_path=True)
            word2vec_model_path = "./word2vec/word2vec-google-news-300.gz"
            w2v = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)
            max_length = 0
            for token_list in X_train:
                np_list = []
                for token in token_list:
                    try:
                        np_list.append(w2v[token])
                    except:
                        continue
                max_length = max(max_length, len(np_list))
                embed_list.append(np.array(np_list))
            embed_list = [np.concatenate([array, np.zeros((max_length-array.shape[0], 300))], axis=0) for array in embed_list]
            return np.array(embed_list)  # [samples, max_length, 300]

        elif self.glove:
            max_length = max([len(token_list) for token_list in X_train])
            glove_model_path = './GloVe/glove2word2vec_model.sav'  # glove embed dimension=100
            glove = pickle.load(open(glove_model_path, 'rb'))
            for token_list in X_train:
                np_list = []
                for token in token_list:
                    try:
                        v = glove[token]
                    except:
                        continue
                    np_list.append(v)
                embed_list.append(
                    np.concatenate([np.array(np_list), np.zeros((max_length - len(token_list), 100))], axis=0))
            embed_array = np.array(embed_list)  # [n_sample, max_length, ndim=512]
            return embed_array

        elif self.elmo: # elmo_dim = 512
            elmo_options_file = "./elmo/elmo_2x2048_256_2048cnn_1xhighway_options.json"
            elmo_weight_file = "./elmo/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"
            elmo_model = Elmo(elmo_options_file, elmo_weight_file, 1, dropout=0)
            character_ids = batch_to_ids(X_train)
            embeddings = elmo_model(character_ids)['elmo_representations'][0]
            return embeddings.detach().numpy()  # [n_sample, max_length, ndim=512]

        elif self.bert:
            pass
        

test = [['ok', ',', 'fine', 'i', 'will', 'check', 'it', 'later', '.'],
        ['love', 'you', '!'],
        ['i', 'think', 'data', 'preprocessing', 'is', 'so', 'complicated', '.']]
embed_model = SciCiteEmbedding(elmo=True)  # pick an embedding method
embed_array = embed_model.embed(test)
np.save(embed_array, '../data/a_simple_test.npy')
print(embed_array.shape)
