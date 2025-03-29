# copied from https://github.com/BlackHC/batchbald_redux/blob/master/batchbald_redux/active_learning.py
import collections
from typing import List

import numpy as np
import torch
import torch.utils.data as data
from sklearn.metrics.pairwise import pairwise_distances_argmin_min


class ActiveLearningData:
    """Splits `dataset` into an active dataset and an available dataset."""

    dataset: data.Dataset
    training_dataset: data.Dataset
    pool_dataset: data.Dataset
    training_mask: np.ndarray
    pool_mask: np.ndarray

    def __init__(self, dataset: data.Dataset):
        super().__init__()
        self.dataset = dataset
        self.training_mask = np.full((len(dataset),), False)
        self.pool_mask = np.full((len(dataset),), True)

        self.training_dataset = data.Subset(self.dataset, None)
        self.pool_dataset = data.Subset(self.dataset, None)

        self._update_indices()

    def _update_indices(self):
        self.training_dataset.indices = np.nonzero(self.training_mask)[0]
        self.pool_dataset.indices = np.nonzero(self.pool_mask)[0]

    def get_dataset_indices(self, pool_indices: List[int]) -> List[int]:
        """Transform indices (in `pool_dataset`) to indices in the original `dataset`."""
        indices = self.pool_dataset.indices[pool_indices]
        return indices

    def acquire(self, pool_indices):
        """
        Acquire elements from the pool dataset into the training dataset.

        Add them to training dataset & remove them from the pool dataset.
        """
        indices = self.get_dataset_indices(pool_indices)

        self.training_mask[indices] = True
        self.pool_mask[indices] = False
        self._update_indices()

    def remove_from_pool(self, pool_indices):
        """
        Remove from the pool dataset.
        """
        indices = self.get_dataset_indices(pool_indices)

        self.pool_mask[indices] = False
        self._update_indices()

    def get_random_pool_indices(self, size) -> torch.LongTensor:
        assert 0 <= size <= len(self.pool_dataset)
        pool_indices = torch.randperm(len(self.pool_dataset))[:size]
        return pool_indices

    def extract_dataset_from_pool(self, size) -> data.Dataset:
        """
        Extract a dataset randomly from the pool dataset and make those indices unavailable.

        Useful for extracting a validation set.
        """
        return self.extract_dataset_from_pool_from_indices(self.get_random_pool_indices(size))

    def extract_dataset_from_pool_from_indices(self, pool_indices) -> data.Dataset:
        """
        Extract a dataset from the pool dataset and make those indices unavailable.

        Useful for extracting a validation set.
        """
        dataset_indices = self.get_dataset_indices(pool_indices)

        self.remove_from_pool(pool_indices)
        return data.Subset(self.dataset, dataset_indices)


# Random selection of samples with equal number of samples per class
def get_balanced_sample_indices(dataset: data.Dataset, num_classes, n_per_digit=2) -> List[int]:
    """Given `target_classes` randomly sample `n_per_digit` for each of the `num_classes` classes."""
    permed_indices = torch.randperm(len(dataset))

    if n_per_digit == 0:
        return []

    num_samples_by_class = collections.defaultdict(int)
    initial_samples = []

    for _, permed_index in enumerate(permed_indices):
        permed_index = int(permed_index)
        _, label = dataset[permed_index]
        index, target = permed_index, int(label)

        num_target_samples = num_samples_by_class[target]
        if num_target_samples == n_per_digit:
            continue

        initial_samples.append(index)
        num_samples_by_class[target] += 1

        if len(initial_samples) == num_classes * n_per_digit:
            break

    return initial_samples


# Acquisition functionality
def get_top_k_scorers(scores_N, batch_size, uncertainty="entropy"):
    N = len(scores_N)
    batch_size = min(batch_size, N)
    largest = True if uncertainty in ["entropy", "energy"] else False
    candidate_scores, candidate_indices = torch.topk(scores_N, batch_size, largest=largest)
    return candidate_scores.tolist(), candidate_indices.tolist()


def find_acquisition_batch(logits, batch_size, score_function, uncertainty="entropy"):
    scores = score_function(logits)
    return get_top_k_scorers(scores, batch_size=batch_size, uncertainty=uncertainty)


def greedy_coreset_selection(unlabeled_embeddings, labeled_embeddings, batch_size):
    """
    Select samples using the greedy Coreset method.
    
    Args:
        unlabeled_embeddings: Embeddings of unlabeled data (numpy array or torch tensor)
        labeled_embeddings: Embeddings of labeled data (numpy array or torch tensor)
        batch_size: Number of samples to select
    
    Returns:
        Tuple of (candidate_scores, candidate_indices) in the format expected by ActiveLearningData
    """
    # Convert to numpy if they are torch tensors
    if isinstance(unlabeled_embeddings, torch.Tensor):
        unlabeled_embeddings = unlabeled_embeddings.cpu().numpy()
    if isinstance(labeled_embeddings, torch.Tensor):
        labeled_embeddings = labeled_embeddings.cpu().numpy()

    a = unlabeled_embeddings.copy()
    b = labeled_embeddings.copy()

    candidate_indices = []
    candidate_scores = []
    original_indices = np.arange(len(a))

    for _ in range(min(batch_size, len(a))):
        if len(a) == 0:
            break

        # Find distances from each unlabeled point to its closest labeled point
        distances = pairwise_distances_argmin_min(a, b)[1]
        # Select the point with maximum distance (furthest from any labeled point)
        max_distance_index = int(np.argmax(distances))
        # Add this index to our candidates
        candidate_indices.append(int(original_indices[max_distance_index]))

        max_distance = float(distances[max_distance_index])
        candidate_scores.append(max_distance)

        # Add the selected point to the labeled set
        b = np.vstack((b, a[max_distance_index:max_distance_index+1]))
        # Remove the selected point from consideration
        a = np.delete(a, max_distance_index, axis=0)
        original_indices = np.delete(original_indices, max_distance_index)

    return candidate_scores, candidate_indices


class RandomFixedLengthSampler(data.Sampler):
    """
    Sometimes, you really want to do more with little data without increasing the number of epochs.
    This sampler takes a `dataset` and draws `target_length` samples from it (with repetition).
    """

    dataset: data.Dataset
    target_length: int

    def __init__(self, dataset: data.Dataset, target_length: int):
        super().__init__(dataset)
        self.dataset = dataset
        self.target_length = target_length

    def __iter__(self):
        # Ensure that we don't lose data by accident.
        if self.target_length < len(self.dataset):
            return iter(torch.randperm(len(self.dataset)).tolist())

        # Sample slightly more indices to avoid biasing towards start of dataset.
        # Have the same number of duplicates for each sample.
        indices = torch.randperm(self.target_length + (-self.target_length % len(self.dataset)))

        return iter((indices[: self.target_length] % len(self.dataset)).tolist())

    def __len__(self):
        return self.target_length
