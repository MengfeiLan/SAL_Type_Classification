import torch
import numpy as np
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer, max_len):

        self.origin_labels = [label for label in df['label_id']]
        origin_labels = self.origin_labels.copy()
        one_hot_list = covert_list_to_one_hot(origin_labels)
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