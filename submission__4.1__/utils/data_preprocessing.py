import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import contractions
# nltk.download('averaged_perceptron_tagger')
import re
import torch
import numpy as np
from collections import defaultdict
from transformers import *
import matplotlib.pyplot as plt



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


class bert_process:

    def __init__(self, scicite_data=None, confidence_level: float = 1, cite2sentence_percent: float = 0.15,
                 max_len: int = 300,
                 batch_size: int = 1, shuffle: bool = True, pretrained_model_name: str = 'bert-base-uncased',
                 padding: str = 'max_length', repeat: list = [1] * 3):

        self.data = []
        self.scicite_data = scicite_data

        self.confidence_level = confidence_level
        self.cite2sentence_percent = cite2sentence_percent
        self.label_map = {'background': 'Background', 'method': 'Uses', 'result': 'CompareOrContrast'}
        # self.label_map = {'background':'Background','result':'CompareOrContrast'}
        self.section_name_vocab = ['introduction', 'experiment', 'conclusion', 'related work', 'method', 'discussion',
                                   'result', 'background']

        self.max_len = max_len
        self.padding = padding

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        if pretrained_model_name == 'allenai/scibert_scivocab_uncased':
            self.citation_id = torch.tensor(8891)  # id for citation
            self.sep_id = torch.tensor(103)  # id for [SEP] in scibert
        else:
            self.citation_id = torch.tensor(11091)  # id for citation
            self.sep_id = torch.tensor(102)  # id for [SEP] in base-bert
        self.cite_pos = []  # citation pisition

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indexed_input = None
        self.indexed_output = None
        self.output_types2idx = {'Background': 0, 'Uses': 1, 'CompareOrContrast': 2}

        self.mask = None
        self.token_type_ids = None

        self.repeat = repeat

        if self.scicite_data:
            self.clean_add_scicite_data()
        self.repeat_minority()
        self.index_input()
        self.index_output()
        self.make_data_loader()

    """
        # acl-arc data:

        # keys:
        # ['text', 'citing_paper_id', 'cited_paper_id', 'citing_paper_year', 'cited_paper_year', 'citing_paper_title', 'cited_paper_title', 'cited_author_ids', 'citing_author_ids', 'extended_context', 
        # 'section_number', 'section_title', 'intent', 'cite_marker_offset', 'sents_before', 'sents_after', 'cleaned_cite_text', 'citation_id', 'citation_excerpt_index', 'section_name']

        # relevant keys: ['text','extended_context','intent','cleaned_cite_text','section_name']

        # 'section_name': {'introduction' , 'experiments', None , 'conclusion', 'related work', 'method'}

        # 'intent': {'Background', 'Uses', 'CompareOrContrast', 'Extends', 'Motivation', 'Future'}




        # scicite data:

        # keys: ['source', 'citeEnd', 'sectionName', 'citeStart', 'string', 'label', 'label_confidence', 'citingPaperId', 'citedPaperId', 'isKeyCitation', 'id', 'unique_id', 'excerpt_index']

        # relevant keys: ['citeEnd', 'sectionName', 'citeStart', 'string', 'label', 'label_confidence']

        # 'sectionName': 
        # Counter({'Discussion': 1240, 'Introduction': 834, 'Methods': 800, '': 587, 'DISCUSSION': 483, 'Results': 359, '1. Introduction': 349, 'METHODS': 296, 'INTRODUCTION': 263, '4. Discussion': 172, 'RESULTS': 162, 
        # '1 Introduction': 160, 'Background': 101, 'Method': 87, 'Results and discussion': 74, '2. Methods': 60, '1. INTRODUCTION': 54, 'Results and Discussion': 41, 'Methodology': 37, 'Materials and methods': 35, 
        # 'RESULTS AND DISCUSSION': 33, '3. Discussion': 31, 'Materials and Methods': 30, '2 Methods': 25, nan: 19, '4 Experiments': 19, '3. Results': 17, 'Experimental design': 17, 'MATERIALS AND METHODS': 16, 
        # '4. DISCUSSION': 16, '5. Discussion': 16, '4 Discussion': 16, '5 Experiments': 16, 'Implementation': 15, 'Present Address:': 13, '2. Method': 13, '3 Experiments': 12, 'METHOD': 11, '6 Experiments': 11, 
        # '3. Methods': 11, '2 Related Work': 11, 'Experiments': 10, '4. Experiments': 10, '1 INTRODUCTION': 10, '3. Results and discussion': 9, 'Experimental Design': 7, '5. Experiments': 7, '3. Methodology': 7, 
        # '2. METHODS': 7, '1. Background': 7, '2. Results and Discussion': 7, '2. Related Work': 7, '2 Related work': 7, 'METHODOLOGY': 7, 'Discussion and conclusions': 7, 'Technical considerations': 6, '3 Methodology': 6,
        # '4 Implementation': 6, '3.2. Yield and characterisation of ethanol organosolv lignin (EOL)': 6, '3. Gaussian mixture and Gaussian mixture-of-experts': 6, '2 Method': 6, 
        # 'Effects of Discourse Stages on Adjustment of Reference Markers': 6, 'Identification of lysine propionylation in five bacteria': 6, '6. Mitochondrial Lesions': 5, '2.2 Behavior and Mission of Autonomous System': 5,
        # 'Structure and function of the placenta': 5, '5. A comparison with sound generation models': 5, 'Conclusions': 5, 'Role of Holistic Thinking': 5, '1.5 The Mystery of the Sushi Repeats: GABAB(1) Receptor Isoforms': 5, 
        # '3. EXPERIMENTS': 5, 'MBD': 5, '3.1. Depression': 5, 'Implications for Explaining Cooperation: Towards a Classification in Terms of Psychological Mechanisms': 4, 'Interpretation Bias in Social Anxiety': 4, '7 Experiments': 4,
        # '4 EXPERIMENTS': 4, '5.2. Implications for Formation of Crustal Plateau Structures': 4, '7 Numerical experiments': 4, 'Changes in CO2 and O3 Effects Through Time': 4, '3. A REAL OPTION TO INVEST UNDER CHOQUET-BROWNIAN AMBIGUITY': 4,
        # .........


        # 'label': Counter({'background': 4840, 'method': 2294, 'result': 1109})

    """

    def plot_sorted_label_confidence(self):
        plt.plot(sorted([exa.get('label_confidence', 0) for exa in self.scicite_data]))
        plt.show()

    def plot_sorted_citationLength_percentage(self):
        percent = []
        for exa in self.scicite_data:
            try:
                percent.append((exa['citeEnd'] - exa['citeStart']) / len(exa['string']))
            except:
                percent.append(1.0)
        plt.plot(sorted(percent), '*')
        # plt.show()
        return sum(np.array(percent) > 0.15)

    def standardized_section_name(self, scicite_section_name: str):

        corrected = re.sub(r'[^a-zA-Z\s]', '', scicite_section_name).lower()  # only keep letters and whitespace

        for name in self.section_name_vocab:
            res = []
            if name in corrected:
                if len(corrected.split()) <= 5:
                    return corrected
                res.append(name)

        if res:
            return (' and '.join(res))
        return None

    def clean_add_scicite_data(self):  # for scicite
        # conditions:
        # 1. has confidence level and >= 1
        # 2. label: 'result' to 'CompareOrContrast'; 'method' to 'uses'
        # 3. citation len / sentence len <= 0.15; citation to @@citation
        # 4. standardize section name
        for exa in self.scicite_data:
            # high_confi = exa.get('label_confidence', 0) >= self.confidence_level
            high_confi = True
            short_cite = isinstance(exa['citeEnd'], (float, int)) and isinstance(exa['citeStart'], (float, int)) and \
                         (exa['citeEnd'] - exa['citeStart']) / len(exa['string']) <= self.cite2sentence_percent
            has_corresponding_label = self.label_map.get(exa['label'], False)
            if high_confi and short_cite and has_corresponding_label:
                exa['intent'] = self.label_map[exa['label']]
                start, end = int(exa['citeStart']), int(exa['citeEnd'])
                exa['cleaned_cite_text'] = exa['string'][:start] + "@@CITATION" + exa['string'][end:]
                try:
                    exa['section_name'] = self.standardized_section_name(exa['sectionName'])
                except:
                    exa['section_name'] = 'no info'
                self.data.append(exa)

    def repeat_minority(self):
        repeated = []
        for exa in self.data:
            for _ in range(self.repeat[self.output_types2idx[exa['intent']]] - 1):
                repeated.append(exa)
        self.data += repeated

    def index_input(self):
        raw_x = []
        for i, exa in enumerate(self.data):
            text, section_name = re.sub("@@CITATION", "@CITATION@", exa['cleaned_cite_text']), exa['section_name']
            if text is None:
                text = ' '
            # if section_name is None:
            #     section_name = ' '
            raw_x.append('sentence : {} [SEP] section name : {}'.format(text, section_name))

        encoded_x = self.tokenizer(raw_x, padding=self.padding, max_length=self.max_len, return_tensors='pt',
                                   truncation=True)  # dict
        self.indexed_input, self.mask, self.token_type_ids = encoded_x['input_ids'], encoded_x['attention_mask'], \
                                                             encoded_x['token_type_ids']

        self.cite_pos = []
        for i, x_i in enumerate(self.indexed_input):
            for j, ele in enumerate(x_i):
                if ele == self.citation_id:
                    self.cite_pos.append(j)
                if ele == self.sep_id:
                    self.token_type_ids[i, j + 1:] = 1
        self.cite_pos = torch.tensor(self.cite_pos)

    def index_output(self):
        self.indexed_output = np.array([self.output_types2idx[exa['intent']] for exa in self.data], dtype=np.int32)

    def make_data_loader(self):

        dataset = BertDataset(self.indexed_input, self.cite_pos, self.indexed_output, self.mask, self.token_type_ids)
        print(len(self.indexed_input))
        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)


class BertDataset:

    def __init__(self, x, citation_pos, y, mask, token_type_ids):
        self.x = x
        self.y = y
        self.citation_pos = citation_pos
        self.mask = mask
        self.token_type_ids = token_type_ids

    def __getitem__(self, idx):
        # self.y[idx]
        return (self.x[idx], self.citation_pos[idx], self.mask[idx], self.token_type_ids[idx]), self.y[idx]

    def __len__(self):
        return len(self.x)