import gensim
import pickle
import numpy
import numpy as np

print(gensim.__version__)

class SciCiteEmbedding:
    def __init__(self, glove=False, word2vec=False, elmo=False, bert=False):
        self.word2vec =word2vec
        self.glove = glove
        self.elmo = elmo
        self.bert = bert
        self.ndim = None
        self.ndim = []

    def embed(self, X_train):
        embed_list = []
        if self.word2vec:
            pass

        elif self.glove:
            model_path = './glove2word2vec_model.sav'  # dimension=100
            glove = pickle.load(open(model_path, 'rb'))
            for token_list in X_train:
                np_list = [glove[token].reshape(1, -1) for token in token_list]
                embed_list.append(np.concatenate(np_list))
            return embed_list

        elif self.elmo:
            pass

        elif self.bert:
            pass

    def seq_padding(self, X_train): # no need!!!
        embed_list = self.embed(X_train)
        pass

    def save_npy(self, X_train):
        pass


test = [['ok',',','fine','i','will','check','it','later','.'],
           ['love','you','!'],
           ['i', 'think','data','preprocessing','is','so','complicated','.']]

embed_model = SciCiteEmbedding(glove=True)
embed_list = embed_model.embed(test)
print(embed_list)
