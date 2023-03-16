#!/usr/bin/env python.

"""
CS4248 ASSIGNMENT 2 Template

TODO: Modify the variables below.  Add sufficient documentation to cross
reference your code with your writeup.

"""

# Import libraries.  Add any additional ones here.
# Generally, system libraries precede others.
# package import
import pandas as pd
import numpy as np
from numpy import isneginf
import random
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import sys
import re
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
import contractions
from word2number import w2n
import time
np.random.seed(12331)
# np.seterr(divide = 'ignore')
# # !{sys.executable} -m pip install contractions
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')
# TODO: Replace with your Student Number
_STUDENT_NUM = 'A0251155M'

class Model:

    def __init__(self, extra_quote=True, extra_haveq=True, extra_properw=True, extra_capital=True, extra_number=True, extra_question=True, contract=True, lemmatize=True, lowercase=True, stopword=True , ngram=1, method='df_idf', model_type='logistic', word_appear=2, num_BOW=5000, **kwargs):

        self.extra_quote=extra_quote
        self.extra_capital=extra_capital
        self.extra_number=extra_number
        self.extra_question=extra_question
        self.contract = contract
        self.lemmatize = lemmatize
        self.lowercase = lowercase
        self.ngram = ngram # integer 1, 2, 3 etc.
        self.method = method # binary weight, term frequency, tf_idf, word embedding
        self.stopword = stopword
        self.BOW = defaultdict()
        self.idf_t = []
        self.model_type = model_type
        self.stopword_set = ['the', 'a', 'and', 'is', 'be'] #set(stopwords.words("english"))  ['the', 'a', 'and', 'is', 'be']
        self.model_return()
        self.word_appear = word_appear
        self.num_BOW = num_BOW
        self.extra_haveq = extra_haveq
        self.extra_properw = extra_properw
        # this part is for the MLPClassifier to set up the
        # self.mlp_arg = {
        #     'hidden_layer': int(kwargs['hidden_layer']),
        #     'activation': kwargs['activation'],
        #     'max_iter': int(kwargs['max_iter']),
        #     'solver': 'adam'
        # }
        self.weight = {0:3.1, 1:1.4, -1:0.5}
        #print(self.mlp_arg)

        if method == 'df_idf':
            self.df_idf = True
        else:
            self.df_idf = False

    def generate_extra_feature(self, texts):
        extra_feature = []
        # find the number of words that is in the quotation mark

        def count_words_in_quotes(text):
            x = re.findall("(\".+\")", text)
            count=0
            if x is None:
                return 0
            else:
                for i in x:
                    t=i[1:-1]
                    def count_words(text):
                        return len(text.split())
                    count+=count_words(t)
                return count

        # the number of capital words
        def count_capital_words(text):
            #print(list(map(str.isupper,text.split())))
            return sum(map(str.istitle,text.split()))

        # count the number of question mark
        def count_question(text):
            question='?'
            return text.count(question)

        # find all number in the sentence
        def count_number(text):
            count = len(re.findall('[0-9]+', text))
            try: # check if there is a number word in the sentence
                w2n.word_to_num(text)
                count += 1
            except:
                pass
            return count
        def if_quotation(text):
            x = re.findall("(\".+\")", text)
            if x:
                have_quote = 1
            else:
                have_quote = 0
            return have_quote
        def count_np(text): # count the proper noun
            text = nltk.WordPunctTokenizer().tokenize(text)
            count = sum(map(lambda word_tag: True if (word_tag[1] == 'NNP') or (word_tag[1] == 'NNPS')  else False, nltk.pos_tag(text)))
            return count
            #new_doc = list(map(lambda word_tag: lemma.lemmatize(word=word_tag[0], pos=tag_map[word_tag[1][0]]), nltk.pos_tag(new_doc)))


        for text in texts:
            text_extra=[]
            if self.extra_quote: #count_words_in_quotes
                text_extra.append(count_words_in_quotes(text))
            if self.extra_capital: #count_capital_words
                text_extra.append(count_capital_words(text))
            if self.extra_question: #count_question
                text_extra.append(count_question(text))
            if self.extra_number: #count_number
                text_extra.append(count_number(text))
            if self.extra_haveq: #count_number
                text_extra.append(if_quotation(text))
            if self.extra_properw: #count_number
                text_extra.append(count_np(text))
            extra_feature.append(text_extra)
        return np.array(extra_feature)

    def preprocessing(self, documents:list)->list:
        new_documents = []
        for document in documents:
            # lower case (this can be control by the augment lowcase)
            if self.lowercase:
                new_doc = document.lower()
            else:
                new_doc = document

            # contraction
            if self.contract:
                new_doc = contractions.fix(new_doc)
            else:
                new_doc = new_doc

            # remove special character, such as double quotes, punctuation, and possessive pronouns.
            def remove_unwant(text):
                text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
                text = re.sub(r'\<a href', ' ', text)
                text = re.sub(r'&amp;', '', text)
                text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
                text = re.sub(r'<br />', ' ', text)
                text = re.sub(r'\'', ' ', text)
                return text
            new_doc = remove_unwant(new_doc)
            # remove stopwords

            if self.stopword:
                new_doc = ' '.join([word for word in new_doc.split() if word not in self.stopword_set])

            # Tokenize the sentence
            new_doc = nltk.WordPunctTokenizer().tokenize(new_doc)

            # Lemmatize the document, control by augment lemmatize
            # for lemmatizing the corpus it is important to distinguish the part of speech
            tag_map = defaultdict(lambda : wn.NOUN)
            tag_map['J'] = wn.ADJ
            tag_map['V'] = wn.VERB
            tag_map['R'] = wn.ADV
            #print(nltk.pos_tag(new_doc))
            if self.lemmatize:
                lemma = nltk.stem.WordNetLemmatizer()
                new_doc = list(map(lambda word_tag: lemma.lemmatize(word=word_tag[0], pos=tag_map[word_tag[1][0]]), nltk.pos_tag(new_doc)))
            new_documents.append(new_doc)

        return new_documents

    def generate_bow(self, texts):
        # augment ngram is to control how many word to be regard as a feature, for example """We are having fun today"""
        # for 2-gram, BOW will contain
        #vocab_freq
        # maybe can be improved
        temp_BOW = []
        for text in texts:
            for i in range(self.ngram):
                for k in range(len(text)-i):
                    word = ' '.join(text[k:k+i+1])
                    #temp.append(word)
                    temp_BOW.append(word)
        #BOW = np.concatenate(BOW)
        temp_BOW = Counter(temp_BOW)
        # only need the word that appears over a certain time
        temp_BOW = {word: time for word, time in temp_BOW.items() if (time >= self.word_appear)} # and (time <= 10000)
        # how many words should we take, self.num_BOW
        #temp_BOW = sorted(temp_BOW.items(), key = lambda wf:(wf[1], wf[0]))
        # if len(temp_BOW)<=self.num_BOW:
        #     pass
        # else:
        #temp_BOW = set(temp_BOW) # this need to be store
        temp_BOW = list(temp_BOW.keys())
        self.BOW = {word:ix for ix, word in enumerate(temp_BOW)}
    # there are several method that can be consider: Binary Weight (bi_w), Term Frequency (t_f). tf_idf
    def convert_vector(self, texts):
        texts_vector=[]
        if self.method=='bi_w':
            for text in texts:
                text_vector = np.zeros(len(self.BOW))
                for i in range(self.ngram):
                    for k in range(len(text)-i):
                        word = ' '.join(text[k:k+i+1])
                        if word in self.BOW:
                            text_vector[self.BOW[word]] = 1
                texts_vector.append(text_vector)
        elif self.method=='t_f':
            for text in texts:
                text_vector = np.zeros(len(self.BOW))
                for i in range(self.ngram):
                    for k in range(len(text)-i):
                        word = ' '.join(text[k:k+i+1])
                        if word in self.BOW:
                            text_vector[self.BOW[word]] += 1
                texts_vector.append(text_vector)
        else: # tf_idf
            D = len(texts)
            # first count the frequency that appear in each text
            texts_vector = []
            for text in texts:
                text_vector = np.zeros(len(self.BOW))
                for i in range(self.ngram):
                    for k in range(len(text)-i):
                        word = ' '.join(text[k:k+i+1])
                        if word in self.BOW:
                            text_vector[self.BOW[word]] += 1
                text_vector = 1+np.log10(text_vector)
                text_vector[isneginf(text_vector)]=0
                texts_vector.append(text_vector)
            texts_vector = np.array(texts_vector)
            df_t = np.count_nonzero(texts_vector, axis=0)
            if len(self.idf_t) == 0:
                # when the vectorized model already trained, when converting the testing dataset use the idf_t of the training
                self.idf_t = np.log10(D/df_t) # this need to be store
            texts_vector = np.array([text_vector*self.idf_t for text_vector in texts_vector])
        return np.array(texts_vector) # return the vecter that need to be train

    # def text_to_vector(self, texts):
    #     if self.method in ['bi_w', 't_f', 'tf_idf']:
    #         self.generate_bow(texts)
    #         texts_vector = self.convert_vector(texts)
    #     else:
    #         pass

    def model_return(self):
        if self.model_type == 'logistic':
            self.classifier = LogisticRegression(max_iter=700, class_weight={0:8, 1:5, -1:1}, C=0.4)# max_iter=300, class_weight={0:8, 1:5, -1:1}
        elif self.model_type == 'nn':
            #self.classifier = MLPClassifier(hidden_layer_sizes=self.mlp_arg['hidden_layer'], activation=self.mlp_arg['activation'], max_iter=self.mlp_arg['max_iter'], solver=self.mlp_arg['solver'])
            self.classifier = MLPClassifier(hidden_layer_sizes=(200,50,10), activation='relu', max_iter=200, solver='adam', alpha=0.00001)


def train_model(model:Model, X_train, y_train):
    ''' TODO: train your model based on the training data '''
    # first is to preprocessing the data
    X_extra_feature = model.generate_extra_feature(X_train)
    X_train = model.preprocessing(X_train)
    # then vectorize the data, i.e., conducting feature engineering
    model.generate_bow(X_train)
    X_train_vector = model.convert_vector(X_train)
    print('the shape of extra feature', X_extra_feature.shape)
    print('before adding the extra feature', X_train_vector.shape)
    try:
        X_train_vector = np.concatenate((X_train_vector, X_extra_feature), axis=1)
    except:
        pass
    print('after adding the extra feature', X_train_vector.shape)
    model.classifier.fit(X_train_vector,y_train)
    print(model.classifier.score(X_train_vector,y_train))

def predict(model, X_test):
    ''' TODO: make your prediction here '''
    X_extra_feature = model.generate_extra_feature(X_test)
    X_test = model.preprocessing(X_test)
    X_test_vector = model.convert_vector(X_test)
    print('the shape of extra feature', X_extra_feature.shape)
    print('before adding the extra feature', X_test_vector.shape)
    try:
        X_test_vector = np.concatenate((X_test_vector, X_extra_feature), axis=1)
    except:
        pass
    print('after adding the extra feature', X_test_vector.shape)
    y_test_predict = model.classifier.predict(X_test_vector)
    return y_test_predict

def generate_result(test, y_pred, filename):
    ''' generate csv file base on the y_pred '''
    test['Verdict'] = pd.Series(y_pred)
    test.drop(columns=['Text'], inplace=True)
    test.to_csv(filename, index=False)

def main():
    ''' load train, val, and test data '''
    train = pd.read_csv('train.csv')
    X_train = train['Text']
    print(len(X_train))
    y_train = train['Verdict']
    model = Model(extra_quote=False, extra_capital=False, extra_number=True, extra_question=True, extra_haveq=False, extra_properw=True, contract=False, lemmatize=False, lowercase=False, stopword=True , ngram=2, method='t_f', model_type='logistic', word_appear=2)
    #test = ['dog chase cat dog', 'cat chase cat', 'car chase tv', 'dog watch dog tv', 'dog cat sit car']
    train_model(model, X_train, y_train)
    # train_model(model, X_train, y_train)
    #test your model
    y_pred = predict(model, X_train)

    # Use f1-macro as the metric
    score = f1_score(y_train, y_pred, average='macro')
    print('score on validation = {}'.format(score))

    # generate prediction on test data
    test = pd.read_csv('test.csv')
    X_test = test['Text']
    y_pred = predict(model, X_test)
    generate_result(test, y_pred, _STUDENT_NUM + ".csv")

# Allow the main class to be invoked if run as a file.
if __name__ == "__main__":
    main()
