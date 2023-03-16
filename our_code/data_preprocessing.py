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

class Data_Preprocessing:
    def __init__(self, contract=False, lemmatize=False, lowercase=False, stopword=False, stopword_set=None):

        self.contract = contract
        self.lemmatize = lemmatize
        self.lowercase = lowercase
        self.stopword = stopword
        self.stopword_set = stopword_set or stopwords.words('english')
        self.clean_text = []

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
            def delete_citation(text):
                # delete the citation from the string
                text = re.sub(r'\([^%]\)', ' ', text)
                text = re.sub(r'\[.*\]', ' ', text)
                # regex_find_citation = re.compile(r"\(\s?(([A-Za-z\-]+\s)+([A-Za-z\-]+\.?)?,?\s\d{2,4}[a-c]?(;\s)?)+\s?\)|"
                #                                  r"\[(\d{1,3},\s?)+\d{1,3}\]|"
                #                                  r"\[[\d,-]+\]|(\([A-Z][a-z]+, \d+[a-c]?\))|"
                #                                  r"([A-Z][a-z]+ (et al\.)? \(\d+[a-c]?\))|"
                #                                  r"[A-Z][a-z]+ and [A-Z][a-z]+ \(\d+[a-c]?\)]")
                # text = regex_find_citation.sub("", text)
                return text
            def delete_space(text):
                text = re.sub(r'\([^%]\)', ' ', text)
                return text
            new_doc = delete_citation(new_doc)

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

        self.clean_text = new_documents
        return self.clean_text



