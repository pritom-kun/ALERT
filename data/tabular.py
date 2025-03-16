import json
import os

import pickle
import pandas as pd
import torch
import transformers
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch.utils import data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TabularDataset(data.Dataset):
    def __init__(self, data_path, save_dir=None):

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

        encoder_path = os.path.join(save_dir, "label_encoder.pkl")

        if os.path.exists(encoder_path):
            # Load existing encoder
            with open(encoder_path, 'rb') as f:
                encoder = pickle.load(f)
        else:
            # Fit and save the encoder if it doesn't exist
            encoder = LabelEncoder()
            encoder.fit(alldata[['label']])
            with open(encoder_path, 'wb') as f:
                pickle.dump(encoder, f)

        print("Number of categories:", len(encoder.classes_))

        self.data = self._tokenize(alldata['text'].tolist())
        self.targets = torch.from_numpy(encoder.transform(alldata[['label']])).to(torch.int64)

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

def create_tabular_dataset(data_path, seed=0, save_dir="./saves"):
    # Create directory for splits if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Define paths for saving splits
    splits_file = os.path.join(save_dir, f"splits_seed_{seed}.json")

    dataset = TabularDataset(data_path, save_dir)

    # Check if splits already exist
    if os.path.exists(splits_file):
        print(f"Loading existing splits from {splits_file}")
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        train_idx = splits['train_idx']
        test_idx = splits['test_idx']
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)

        for train_idx, test_idx in splitter.split(dataset, dataset.targets.cpu()):
            # Convert indices to Python lists for JSON serialization
            train_idx = train_idx.tolist()
            test_idx = test_idx.tolist()

        # Save splits to disk
        with open(splits_file, 'w') as f:
            json.dump({'train_idx': train_idx, 'test_idx': test_idx}, f)
        print(f"Saved splits to {splits_file}")

    train_dataset = data.Subset(dataset, train_idx)
    test_dataset = data.Subset(dataset, test_idx)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_dataset, test_dataset, dataset.tokenizer
