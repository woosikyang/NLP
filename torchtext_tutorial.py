'''

Creating Dataset

'''

import torch
import torchtext.data as data
import pandas as pd
import os
import pickle

with open('data/data1.txt', 'rb') as f:
    train = pickle.load(f)

train = train.iloc[:,:-1]
train = train[['business_goal','class_code']]
train.columns = ['text','label']


with open('data/data2.txt', 'rb') as f:
    test = pickle.load(f)

test = test.iloc[-5000:,:-1]
test = test[['business_goal','class_code']]
test.columns = ['text','label']

# step1
TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=str.split,
                  lower=True,
                  batch_first=True)

LABEL = data.LabelField()

# step2 - Dataset


class DataFrameDataset(data.Dataset):

    def __init__(self, df, text_field, label_field, is_test=False, **kwargs):
        fields = [('text', text_field), ('label', label_field)]
        examples = []
        for i, row in df.iterrows():
            label = row.label if not is_test else None
            text = row.text
            examples.append(data.Example.fromlist([text, label], fields))

        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, text_field, label_field, train_df, val_df=None, test_df=None, **kwargs):
        train_data, val_data, test_data = (None, None, None)

        if train_df is not None:
            train_data = cls(train_df.copy(), text_field, label_field, **kwargs)
        if val_df is not None:
            val_data = cls(val_df.copy(), text_field, label_field, **kwargs)
        if test_df is not None:
            test_data = cls(test_df.copy(), text_field, label_field, True, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)

train_ds, test_ds = DataFrameDataset.splits(
    text_field=TEXT, label_field=LABEL, train_df=train, test_df=test)


TEXT.build_vocab(train_ds)
len(TEXT.vocab)
