import json

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch.utils import data
from torch.utils.data import Sampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TabularDataset(data.Dataset):
    def __init__(self, data_path):

        with open(data_path) as f:
            data_json = json.loads(f.read())

        alldata = pd.DataFrame(
            [
                {'text': row['text'], 'label': row['mappings'][0]['attack_id']}
                for row in data_json['sentences']
                if len(row['mappings']) > 0
            ]
        )

        self.tokenizer = transformers.BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", max_length=512)

        encoder = LabelEncoder()
        encoder.fit(alldata[['label']])

        print("Number of categories:", len(encoder.classes_))

        self.data = self._tokenize(alldata['text'].tolist())
        self.targets = torch.from_numpy(encoder.transform(alldata[['label']])).to(torch.int64)

        # print(self.targets)

        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data.to(device), self.targets.to(device)


    def _tokenize(self, samples: 'list[str]'):
        return self.tokenizer(samples, return_tensors='pt', padding='max_length', truncation=True, max_length=512).input_ids


    def __len__(self):
        # Return the number of samples in the dataset
        return self.data.size(0)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        feat, target = self.data[index], self.targets[index]

        return feat, target

def create_tabular_dataset(data_path, seed=0):

    dataset = TabularDataset(data_path)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)

    for train_idx, test_idx in splitter.split(dataset, dataset.targets.cpu()):
        train_dataset = data.Subset(dataset, train_idx)
        test_dataset = data.Subset(dataset, test_idx)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_dataset, test_dataset, dataset.tokenizer
