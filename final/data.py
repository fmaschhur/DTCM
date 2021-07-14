from typing import List

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from transformers import BertTokenizer


def tokenize_sentences(sentences: List[str], bert_base_model: str):
    bert_tokenizer = BertTokenizer.from_pretrained(bert_base_model)
    return bert_tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")


def get_test_dl(eval_data_file_path: str, bert_base_model: str):
    eval_data = pd.read_csv(eval_data_file_path)

    sent_ids = list(eval_data['sent_id'])
    sentences = list(eval_data['sentence'])

    sentences_tokenized = tokenize_sentences(sentences, bert_base_model)

    test_data_dict = {'id': sent_ids, 'st': sentences_tokenized}
    dataset = SentenceComplexityDataset(test_data_dict)

    return DataLoader(dataset)


class SentenceComplexityDataset(Dataset):
    def __init__(self, data):
        self.st = data['st']
        self.id = Tensor(data['id'])

    # number of rows in the dataset
    def __len__(self):
        return len(self.id)

    # get a row at an index
    def __getitem__(self, idx):
        item = {key: val[idx, :] for key, val in self.st.items()}
        item['id'] = self.id[idx]
        return item

