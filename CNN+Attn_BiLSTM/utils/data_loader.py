from utils.data_preprocessing import DataPreprocessing, DataReader
from torchtext.legacy.data import Field, LabelField, Example, Dataset
from torchtext.legacy.data import Iterator, BucketIterator
import pandas as pd

# return batch iterators for training
def get_iters(batch_size, use_bert=False, use_balance_data=True, train_data_path="./scicite-data/train.jsonl", valid_data_path="./scicite-data/dev.jsonl", test_data_path="./scicite-data/test.jsonl"):
    dp = DataPreprocessing(contract=True, lemmatize=False, lowercase=True, stopword=False, stopword_set=None)
    if use_balance_data:
        train_data = pd.read_csv("./scicite-data/train_balanced.csv")
        train_data.drop_duplicates(['string'])
    else:
        train_data = DataReader(train_data_path).read()
    valid_data = DataReader(valid_data_path).read()
    valid_data['string'] = dp.preprocessing(list(valid_data['string']))

    test_data = DataReader(test_data_path).read()
    test_data['string'] = dp.preprocessing(list(test_data['string']))

    # construct Fields
    # tokenizer: not needed
    TEXT = Field(sequential=True, tokenize=None, batch_first=True)
    LABEL = LabelField()

    train_examples, train_fields = get_dataset(train_data, TEXT, LABEL)
    valid_examples, valid_fields = get_dataset(valid_data, TEXT, LABEL)
    test_examples, test_fields = get_dataset(test_data, TEXT, None, test=True)

    train = Dataset(train_examples, train_fields)
    valid = Dataset(valid_examples, valid_fields)
    test = Dataset(test_examples, test_fields)

    if use_bert:
        TEXT.build_vocab(train)  # bert: no not need to embed at this stage
    else:
        TEXT.build_vocab(train, vectors="glove.6B.100d")  # start embedding, download vectors
    LABEL.build_vocab(train)
    print('LABEL.vocab.stoi:', LABEL.vocab.stoi)  # defaultdict(None, {'background': 0, 'method': 1, 'result': 2})
    # print vocab information
    print('len(TEXT.vocab)', len(TEXT.vocab))
    if not use_bert:
        print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

    # construct iterators
    train_iter, val_iter = BucketIterator.splits(
        (train, valid),
        batch_sizes=(batch_size, batch_size),
        device=-1,  # device=-1 if not using gpu
        sort_key=lambda x: len(x.string),  # the BucketIterator needs to be told what function it should use to
        # group the data.
        sort_within_batch=False,
        repeat=False  # we pass repeat=False because we want to wrap this Iterator layer.
    )
    test_iter = Iterator(test, batch_size=batch_size, device=-1, sort=False, sort_within_batch=False, repeat=False)

    # for idx, batch in enumerate(train_iter):  # output batch shapes
    #     print(idx, batch)
    #     text, label = batch.string, batch.label
    #     print(text.shape, label.shape)
    #     print('text[0]', text[0, :])
    #     if idx == 2:
    #         break
    return train_iter, val_iter, test_iter, TEXT, LABEL

