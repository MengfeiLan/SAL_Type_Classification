import torch
import numpy as np
from transformers import BertTokenizer
from utils import *
from torch.optim import Adam
import torch
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.nn import functional as F
from torch.utils.data._utils.collate import default_collate

def my_collate_fn(batch):
    elem = batch[0]

    return_dict = {}
    for key in elem:
        if key == "encoded_tgt_text":
            return_dict[key] = [d[key] for d in batch]
        else:
            try:
                return_dict[key] = default_collate([d[key] for d in batch])
            except:
                return_dict[key] = [d[key] for d in batch]

    return return_dict

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer, max_len, num_label):

        self.origin_labels = [label for label in df['label_id']]
        origin_labels = self.origin_labels.copy()
        one_hot_list = covert_list_to_one_hot(origin_labels, num_label)
        self.labels = one_hot_list
        self.tokenizer = tokenizer
        self.texts = [self.tokenizer(text,
                               padding='max_length', max_length = max_len, truncation=True,
                                return_tensors="pt") for text in df['sentences']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels

        return np.array(self.labels[idx])

    def get_origin_batch_labels(self, idx):
        # Fetch a batch of labels

        return np.array(self.origin_labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        batch_texts["labels"] = batch_y
        batch_texts["origin_labels"] = self.get_origin_batch_labels(idx)
        
        return batch_texts
