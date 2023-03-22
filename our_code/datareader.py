import json
import pandas as pd


class DataReader:
    def __init__(self, data_path):
        self.data_path = data_path

    def read(self):
        # load data
        original_data = [json.loads(line) for line in open(self.data_path,encoding='utf-8')]
        original_data = pd.json_normalize(original_data)
        # delete columns
        remain_list = ['string', 'label', 'label_confidence']
        data = original_data[remain_list]
        # drop duplicates
        data.drop_duplicates(['string'])
        # return data
        return data
