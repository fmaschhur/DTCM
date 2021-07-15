from typing import List, Dict

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from transformers import BertTokenizer, BatchEncoding


def tokenize_sentences(sentences: List[str], bert_base_model: str) -> BatchEncoding:
    """Tokenizes and pads a list of sentences using the model specified.

    Parameters
    ----------
    sentences : List[str]
        a list containing all sentences that are to be tokenized
        in this function
    bert_base_model : str
        specifies which huggingface transformers model is to be
        used for the tokenization 
    
    Returns
    -------
    BatchEncoding
        huggingface transformers data structure which contains
        the padded and tokenized sentences
    
    """

    bert_tokenizer = BertTokenizer.from_pretrained(bert_base_model)
    return bert_tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")


def get_test_dl(eval_data_file_path: str, bert_base_model: str) -> DataLoader:
    """Loads a dataset file into a DataLoader object.
    
    A given CSV dataset file is read and tokenized. The processed
    contents are then loaded into a pytorch DataLoader object,
    which is returned in the end. 

    Parameters
    ----------
    eval_data_file_path : str
        the path to a valid CSV dataset file.
        Check the README for more informaiton 
    bert_base_model : str
        specifies which huggingface transformers model is to be
        used for the tokenization 
    
    Returns
    -------
    DataLoader
        pytorch DataLoader object which contains the padded and
        tokenized sentences
    
    """

    eval_data = pd.read_csv(eval_data_file_path)

    # retrieving the contents of the dataset
    sent_ids = list(eval_data['sent_id'])
    sentences = list(eval_data['sentence'])

    sentences_tokenized = tokenize_sentences(sentences, bert_base_model)

    # saving the ids and tokenized sentences into the according dataset object
    test_data_dict = {'id': sent_ids, 'st': sentences_tokenized}
    dataset = SentenceComplexityDataset(test_data_dict)

    return DataLoader(dataset)


class SentenceComplexityDataset(Dataset):
    """
    A dataset class used to load the contents of a CSV
    dataset file as specified by the README.

    ...

    Attributes
    ----------
    st : BatchEncoding
        huggingface transformers data structure which contains
        the padded and tokenized sentences
    id : Tensor
        a Tensor containing all the sentence ids related to the
        padded and tokenized sentences in st
    
    Methods
    -------
    __len__()
        returns the length of the dataset
    __getitem__(idx)
        returns the tokenized sentence and the id of the
        sentence at index idx
    """

    def __init__(self, data):
        """
        Parameters
        ----------
        data : Dict
            a dict containing both the padded and tokenized
            sentences and the ids of those sentences
        """

        self.st = data['st']
        self.id = Tensor(data['id'])

    # number of rows in the dataset
    def __len__(self) -> int:
        """returns the length of the dataset"""
        return len(self.id)

    # get a row at an index
    def __getitem__(self, idx) -> Dict:
        """returns the tokenized sentence and the id of the sentence at index idx
        
        Parameters
        ----------
        idx : int
            specifies the index of the data to be returned
        """

        item = {key: val[idx, :] for key, val in self.st.items()}
        item['id'] = self.id[idx]
        return item

