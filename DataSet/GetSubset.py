import torch
from torch.utils.data import Subset
from torch.utils.data import Dataset


def get_subset(dataset: Dataset, split_index: int) -> (Subset, Subset):
    lower_indices = torch.arange(0, split_index, step=1)
    upper_indices = torch.arange(split_index, len(dataset), step=1)

    lower_set = Subset(dataset, lower_indices)
    upper_set = Subset(dataset, upper_indices)

    return lower_set, upper_set
