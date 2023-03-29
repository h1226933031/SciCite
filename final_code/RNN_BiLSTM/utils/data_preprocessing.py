import nltk
# nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import re
from collections import defaultdict
import contractions
import json
import pandas as pd


class DataPreprocessing:
    def __init__(self, contract=False, lemmatize=False, lowercase=False, stopword=False, stopword_set=None):

        self.contract = contract
        self.lemmatize = lemmatize
        self.lowercase = lowercase
        self.stopword = stopword
        self.stopword_set = stopword_set or stopwords.words('english')
        self.clean_text = []

    def preprocessing(self, documents: list) -> list:
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
            tag_map = defaultdict(lambda: wn.NOUN)
            tag_map['J'] = wn.ADJ
            tag_map['V'] = wn.VERB
            tag_map['R'] = wn.ADV
            # print(nltk.pos_tag(new_doc))
            if self.lemmatize:
                lemma = nltk.stem.WordNetLemmatizer()
                new_doc = list(map(lambda word_tag: lemma.lemmatize(word=word_tag[0], pos=tag_map[word_tag[1][0]]),
                                   nltk.pos_tag(new_doc)))
            new_documents.append(new_doc)

        self.clean_text = new_documents
        return self.clean_text


class DataReader:
    def __init__(self, data_path):
        self.data_path = data_path

    def read(self):
        # load data
        original_data = [json.loads(line) for line in open(self.data_path, encoding='utf-8')]
        original_data = pd.json_normalize(original_data)
        # delete columns
        remain_list = ['string', 'label']  # , 'label_confidence']
        data = original_data[remain_list]
        # drop duplicates
        data.drop_duplicates(['string'])
        # return data
        return data
