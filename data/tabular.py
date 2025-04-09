import json
import os
import pickle

import nlpaug.augmenter.word as naw
import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch.utils import data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TabularDataset(data.Dataset):
    def __init__(self, data_path, save_dir=None, transform=False):

        self.transform = transform

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

        if self.transform:
            # First augmentation with insert
            aug = naw.ContextualWordEmbsAug(
                model_path='bert-base-uncased', action="insert", device='cuda', top_k=10, aug_max=5, batch_size=16)
            self.data_aug_ins = self._tokenize(aug.augment(alldata['text'].tolist()))

            # Second augmentation with substitute
            aug = naw.ContextualWordEmbsAug(
                model_path='bert-base-uncased', action="substitute", device='cuda', top_k=10, aug_max=5, batch_size=16)
            self.data_aug_sub = self._tokenize(aug.augment(alldata['text'].tolist()))

            print("Augmentation done")

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

        if self.transform:
            # Return augmented data
            feat_aug_ins = self.data_aug_ins[index]
            feat_aug_sub = self.data_aug_sub[index]
            return feat, feat_aug_ins, feat_aug_sub, target

        else:
            return feat, target


def create_tabular_dataset(data_path, seed=0, save_dir="./saves", transform=False):
    # Create directory for splits if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Define paths for saving splits
    splits_file = os.path.join(save_dir, f"splits_seed_{seed}.json")

    dataset = TabularDataset(data_path, save_dir, transform)

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


def create_tabular_ood_dataset(data_path, num_id_classes=40, seed=0, save_dir="./saves"):
    """
    Create dataset with left-out classes for OOD detection.
    
    Args:
        data_path: Path to the data
        ood_ratio: Fraction of classes to hold out as OOD
        seed: Random seed
        save_dir: Directory to save splits and encoder
    """
    # Create directory for splits if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Define paths for saving splits
    splits_file = os.path.join(save_dir, f"ood_splits_seed_{seed}_ratio_{0.2}.json")

    dataset = TabularDataset(data_path, save_dir)
    
    # Get all unique classes
    all_classes = torch.unique(dataset.targets).cpu().numpy()
    all_classes_sorted = sorted([int(x) for x in all_classes])

    # Check if splits already exist
    if os.path.exists(splits_file):
        print(f"Loading existing splits from {splits_file}")
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        id_classes = splits['id_classes']
        ood_classes = splits['ood_classes']
        train_idx = splits['train_idx']
        id_test_idx = splits['id_test_idx']
        ood_test_idx = splits['ood_test_idx']
    else:
        # Randomly select classes for OOD
        np.random.seed(seed)
        id_classes = all_classes_sorted[:num_id_classes]
        ood_classes = all_classes_sorted[num_id_classes:]

        # Create masks for ID and OOD samples
        targets_np = dataset.targets.cpu().numpy()
        id_mask = np.isin(targets_np, id_classes)
        ood_mask = np.isin(targets_np, ood_classes)

        id_indices = np.where(id_mask)[0]
        ood_indices = np.where(ood_mask)[0]

        # Split ID data into train and test
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        id_train_indices, id_test_indices = next(splitter.split(
            id_indices, targets_np[id_indices]
        ))

        # Get actual indices
        train_idx = id_indices[id_train_indices].tolist()
        id_test_idx = id_indices[id_test_indices].tolist()

        # Use all OOD data as test set (no pool split)
        ood_test_idx = ood_indices.tolist()

        # Save splits to disk
        with open(splits_file, 'w') as f:
            json.dump({
                'id_classes': [int(x) for x in id_classes],
                'ood_classes': [int(x) for x in ood_classes],
                'train_idx': train_idx,
                'id_test_idx': id_test_idx,
                'ood_test_idx': ood_test_idx
            }, f)
        print(f"Saved splits to {splits_file}")

    # Create the datasets
    train_dataset = data.Subset(dataset, train_idx)
    id_test_dataset = data.Subset(dataset, id_test_idx)
    ood_test_dataset = data.Subset(dataset, ood_test_idx)

    print(f"In-distribution classes: {len(id_classes)}")
    print(f"Out-of-distribution classes: {len(ood_classes)}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"ID test dataset size: {len(id_test_dataset)}")
    print(f"OOD test dataset size: {len(ood_test_dataset)}")

    return train_dataset, id_test_dataset, ood_test_dataset, dataset.tokenizer
