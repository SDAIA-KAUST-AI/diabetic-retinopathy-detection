import os
from typing import (Dict, Optional, Tuple,
                    Union, Callable, Iterable)
import pandas as pd
from PIL import Image
from enum import Enum
import numpy as np
from numpy.random import RandomState
import collections.abc
from collections import Counter, defaultdict

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader


from src.labelmap import DR_LABELMAP


DataRecord = Tuple[Image.Image, int]


class RetinopathyDataset(data.Dataset[DataRecord]):
    """ A class to access the pre-downloaded Diabetic Retinopathy dataset. """

    def __init__(self, data_path: str) -> None:
        """ Constructor.

        Args:
            data_path (str): path to the dataset, ex: "retinopathy_data"
                containing "trainLabels.csv" and "train/".
        """
        super().__init__()

        self.data_path = data_path

        self.ext = ".jpeg"

        anno_path = os.path.join(data_path, "trainLabels.csv")
        self.anno_df = pd.read_csv(anno_path) # ['image', 'level']
        anno_name_set = set(self.anno_df['image']) 

        if True:
            train_path = os.path.join(data_path, "train")
            img_path_list = os.listdir(train_path)
            img_name_set = set([os.path.splitext(p)[0] for p in img_path_list])
            assert anno_name_set == img_name_set

        self.label_map = DR_LABELMAP
    
    def __getitem__(self, index: Union[int, slice]) -> DataRecord:
        assert isinstance(index, int)
        img_path = self.get_path_at(index)
        img = Image.open(img_path)
        label = self.get_label_at(index)
        return img, label

    def __len__(self) -> int:
        return len(self.anno_df)
    
    def get_label_at(self, index: int) -> int:
        label = self.anno_df['level'].iloc[index].item()
        return label

    def get_path_at(self, index: int) -> str:
        img_name = self.anno_df['image'].iloc[index]
        img_path = os.path.join(self.data_path, "train", img_name+self.ext)
        return img_path


""" Purpose of a split: training or validation. """
class Purpose(Enum):
    Train = 0
    Val = 1

""" Augmentation transformations for an image and a label. """
FeatureAndTargetTransforms = Tuple[Callable[..., torch.Tensor],
                                   Callable[..., torch.Tensor]]

""" Feature (image) and target (label) tensors. """
TensorRecord = Tuple[torch.Tensor, torch.Tensor]


class Split(data.Dataset[TensorRecord], collections.abc.Sequence[TensorRecord]):
    """ Split is a class that keep a view on a part of a dataset.
    Split is used to hold the imormation about which samples go to training
    and which to validation without a need to put these groups of files into
    separate folders.
    """
    def __init__(self, dataset: RetinopathyDataset,
                 indices: np.ndarray,
                 purpose: Purpose,
                 transforms: FeatureAndTargetTransforms,
                 oversample_factor: int = 1,
                 stratify_classes: bool = False,
                 use_log_frequencies: bool = False,
                 ):
        """ Constructor.

        Args:
            dataset (RetinopathyDataset): The dataset on which the Split "views".
            indices (np.ndarray): Externally provided indices of samples that
                are "viewed" on.
            purpose (Purpose): Either train or val, to be able to replicate
                the data for train split for effecient workers utilization.
            transforms (FeatureAndTargetTransforms): Functors of feature and
                target transforms.
            oversample_factor (int, optional): Expand the training dataset by
                replication to avoid dataloader stalls on epoch ends. Defaults to 1.
            stratify_classes (bool, optional): Whether to apply stratified sampling.
                Defaults to False.
            use_log_frequencies (bool, optional): If stratify_classes=True,
                whether to use logarithmic sampling strategy. If False, apply
                regular even sampling. Defaults to False.
        """
        self.dataset = dataset
        self.indices = indices
        self.purpose = purpose
        self.feature_transform = transforms[0]
        self.target_transform = transforms[1]
        self.oversample_factor = oversample_factor
        self.stratify_classes = stratify_classes
        self.use_log_frequencies = use_log_frequencies

        self.per_class_indices: Optional[Dict[int, np.ndarray]] = None
        self.frequencies: Optional[Dict[int, float]] = None
        if self.stratify_classes:
            self._bucketize_indices()
            if self.use_log_frequencies:
                self._calc_frequencies()

    def _calc_frequencies(self):
        assert self.per_class_indices is not None
        counts_dict = {lbl: len(arr) for lbl, arr in self.per_class_indices.items()}
        counts = np.array(list(counts_dict.values()))
        counts_nrm = self._normalize(counts)
        temperature = 50.0 # > 1 to even-out frequencies
        freqs = self._normalize(np.log1p(counts_nrm * temperature))
        self.frequencies = {k: freq.item() for k, freq
                            in zip(self.per_class_indices.keys(), freqs)}
        print(self.frequencies)

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        return arr / np.sum(arr)

    def _bucketize_indices(self):
        buckets = defaultdict(list)
        for index in self.indices:
            label = self.dataset.get_label_at(index)
            buckets[label].append(index)
        self.per_class_indices = {k: np.array(v)
                                  for k, v in buckets.items()}

    def __getitem__(self, index: Union[int, slice]) -> TensorRecord: # type: ignore[override]
        assert isinstance(index, int)
        if self.purpose == Purpose.Train:
            index_rem = index % len(self.indices)
            idx = self.indices[index_rem].item()
        else:
            idx = self.indices[index].item()
        if self.per_class_indices:
            if self.frequencies is not None:
                arange = np.arange(len(self.per_class_indices))
                frequencies = np.zeros(len(self.per_class_indices), dtype=float)
                for k, v in self.frequencies.items():
                    frequencies[k] = v
                random_key = np.random.choice(
                    arange,
                    p=frequencies)
            else:
                random_key = np.random.randint(len(self.per_class_indices))

            indices = self.per_class_indices[random_key]
            actual_index = np.random.choice(indices).item()
        else:
            actual_index = idx
        feature, target = self.dataset[actual_index]
        feature_tensor = self.feature_transform(feature)
        target_tensor = self.target_transform(target)
        return feature_tensor, target_tensor

    def __len__(self):
        if self.purpose == Purpose.Train:
            return len(self.indices) * self.oversample_factor
        else:
            return len(self.indices)

    @staticmethod
    def make_splits(all_data: RetinopathyDataset,
                    train_transforms: FeatureAndTargetTransforms,
                    val_transforms: FeatureAndTargetTransforms,
                    train_fraction: float,
                    stratify_train: bool,
                    stratify_val: bool,
                    seed: int = 54,
                    ) -> Tuple['Split', 'Split']:

        """ Prepare train and val splits deterministically.

        Returns:
            Tuple[Split, Split]:
                - Train split
                - Val split
        """

        prng = RandomState(seed)

        num_train = int(len(all_data) * train_fraction)
        all_indices = prng.permutation(len(all_data))
        train_indices = all_indices[:num_train]
        val_indices = all_indices[num_train:]
        train_data = Split(all_data, train_indices, Purpose.Train,
                           train_transforms, stratify_classes=stratify_train)
        val_data = Split(all_data, val_indices, Purpose.Val,
                         val_transforms, stratify_classes=stratify_val)
        return train_data, val_data


def print_data_stats(dataset: Union[Iterable[DataRecord], DataLoader],
                     split_name: str) -> None:
    labels = []
    for _, label in dataset:
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()
        labels.append(label)
    labels = np.concatenate(labels)
    cnt = Counter(labels)
    print(cnt)


