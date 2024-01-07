import os
from typing import (Any, List, Dict, Optional, Tuple,
                    Union, Callable, Iterable, Iterator)
import pandas as pd
from PIL import Image
import datetime
from argparse import ArgumentParser
from enum import Enum
import numpy as np
from numpy.random import RandomState
import collections.abc
from collections import Counter, defaultdict
import math

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader

from torchvision.transforms import (
    CenterCrop, 
    Compose, 
    Normalize, 
    RandomHorizontalFlip,
    RandomResizedCrop, 
    RandomRotation,
    RandomAffine,
    Resize, 
    ToTensor)

from transformers import ViTImageProcessor
from transformers import ViTForImageClassification
from transformers import AdamW

from transformers import AutoImageProcessor, ResNetForImageClassification

import lightning as L
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelSummary
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification.accuracy import MulticlassAccuracy
from torchmetrics.classification import MulticlassCohenKappa

from labelmap import DR_LABELMAP


DataRecord = Tuple[Image.Image, int]


class RetinopathyDataset(data.Dataset[DataRecord]):
    def __init__(self, data_path: str) -> None:
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


class Purpose(Enum):
    Train = 0
    Val = 1


FeatureAndTargetTransforms = Tuple[Callable[..., torch.Tensor],
                                   Callable[..., torch.Tensor]]

TensorRecord = Tuple[torch.Tensor, torch.Tensor]

def normalize(arr: np.ndarray) -> np.ndarray:
    return arr / np.sum(arr)


class Split(data.Dataset[TensorRecord], collections.abc.Sequence[TensorRecord]):
    def __init__(self, dataset: RetinopathyDataset,
                 indices: np.ndarray,
                 purpose: Purpose,
                 transforms: FeatureAndTargetTransforms,
                 oversample_factor: int = 1,
                 stratify_classes: bool = False,
                 use_log_frequencies: bool = False,
                 ):

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
            self.bucketize_indices()
            if self.use_log_frequencies:
                self.calc_frequencies()

    def calc_frequencies(self):
        assert self.per_class_indices is not None
        counts_dict = {lbl: len(arr) for lbl, arr in self.per_class_indices.items()}
        counts = np.array(list(counts_dict.values()))
        counts_nrm = normalize(counts)
        temperature = 50.0 # > 1 to even-out frequencies
        freqs = normalize(np.log1p(counts_nrm * temperature))
        self.frequencies = {k: freq.item() for k, freq
                            in zip(self.per_class_indices.keys(), freqs)}
        print(self.frequencies)

    def bucketize_indices(self):
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


def print_data_stats(dataset: Union[Iterable[DataRecord], DataLoader], split_name: str) -> None:
    labels = []
    for _, label in dataset:
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()
        labels.append(label)
    labels = np.concatenate(labels)
    cnt = Counter(labels)
    print(cnt)


class Metrics:
    def __init__(self,
                    num_classes: int,
                    labelmap: Dict[int, str],
                    split: str,
                    log_fn: Callable[..., None]) -> None:
        self.labelmap = labelmap
        self.loss = MeanMetric(nan_strategy='ignore')
        self.accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.per_class_accuracies = MulticlassAccuracy(
            num_classes=num_classes, average=None)
        self.kappa = MulticlassCohenKappa(num_classes)
        self.split = split
        self.log_fn = log_fn
    
    def update(self,
               loss: torch.Tensor,
               preds: torch.Tensor,
               labels: torch.Tensor) -> None:
        self.loss.update(loss)
        self.accuracy.update(preds, labels)
        self.per_class_accuracies.update(preds, labels)
        self.kappa.update(preds, labels)

    def log(self) -> None:
        loss = self.loss.compute()
        accuracy = self.accuracy.compute()
        accuracies = self.per_class_accuracies.compute()
        kappa = self.kappa.compute()
        mean_accuracy = torch.nanmean(accuracies)
        self.log_fn(f"{self.split}/loss", loss, sync_dist=True)
        self.log_fn(f"{self.split}/accuracy", accuracy, sync_dist=True)
        self.log_fn(f"{self.split}/mean_accuracy", mean_accuracy, sync_dist=True)
        for i_class, acc in enumerate(accuracies):
            name = self.labelmap[i_class]
            self.log_fn(f"{self.split}/acc/{i_class} {name}", acc, sync_dist=True)
        self.log_fn(f"{self.split}/kappa", kappa, sync_dist=True)

    def to(self, device) -> 'Metrics':
        self.loss.to(device) # BUG HERE? should I assign it back?
        self.accuracy.to(device)
        self.per_class_accuracies.to(device)
        self.kappa.to(device)
        return self


def worker_init_fn(worker_id):
    state = np.random.get_state()
    assert isinstance(state, tuple)
    assert isinstance(state[1], np.ndarray)
    seed_arr = state[1]
    seed_np = seed_arr[0] + worker_id
    np.random.seed(seed_np)
    seed_pt = seed_np + 1111
    torch.manual_seed(seed_pt)
    print(f"Setting numpy seed to {seed_np} and pytorch seed to {seed_pt} in worker {worker_id}")


class ViTLightningModule(L.LightningModule):
    def __init__(self, debug: bool) -> None:
        super().__init__()

        self.save_hyperparameters()

        np.random.seed(53)

        # pretrained_name = 'google/vit-base-patch16-224-in21k'
        # pretrained_name = 'google/vit-base-patch16-384-in21k'

        # pretrained_name = "microsoft/resnet-50"
        pretrained_name = "microsoft/resnet-34"

        # processor = ViTImageProcessor.from_pretrained(pretrained_name)
        processor = AutoImageProcessor.from_pretrained(pretrained_name)

        image_mean = processor.image_mean # type: ignore
        image_std = processor.image_std # type: ignore
        # size = processor.size["height"] # type: ignore
        # size = processor.size["shortest_edge"] # type: ignore
        size = 896 # 448

        normalize = Normalize(mean=image_mean, std=image_std)
        train_transforms = Compose(
            [
                # RandomRotation((-180, 180)),
                RandomAffine((-180, 180), shear=10),
                RandomResizedCrop(size, scale=(0.5, 1.0)),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )
        val_transforms = Compose(
            [
                Resize(size),
                CenterCrop(size),
                ToTensor(),
                normalize,
            ]
        )

        self.dataset = RetinopathyDataset("retinopathy_data")

        # print_data_stats(self.dataset, "all_data")

        train_data, val_data = Split.make_splits(
            self.dataset,
            train_transforms=(train_transforms, torch.tensor),
            val_transforms=(val_transforms, torch.tensor),
            train_fraction=0.9,
            stratify_train=True,
            stratify_val=True,
            )

        assert len(set(train_data.indices).intersection(set(val_data.indices))) == 0

        label2id = {label: id for id, label in self.dataset.label_map.items()}

        num_classes = len(self.dataset.label_map)
        labelmap = self.dataset.label_map
        assert len(labelmap) == num_classes
        assert set(labelmap.keys()) == set(range(num_classes))

        train_batch_size = 4 if debug else 20
        val_batch_size = 4 if debug else 20

        num_gpus = torch.cuda.device_count()
        print(f"{num_gpus=}")

        num_cores = torch.get_num_threads()
        print(f"{num_cores=}")

        num_threads_per_gpu = max(1, int(math.ceil(num_cores / num_gpus))) \
            if num_gpus > 0 else 1

        num_workers = 1 if debug else num_threads_per_gpu
        print(f"{num_workers=}")

        self._train_dataloader = DataLoader(
            train_data,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=True,
            batch_size=train_batch_size,
            worker_init_fn=worker_init_fn,
            )
        self._val_dataloader = DataLoader(
            val_data,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=True,
            batch_size=val_batch_size,
            )

        # print_data_stats(self._val_dataloader, "val")
        # print_data_stats(self._train_dataloader, "train")

        img_batch, label_batch = next(iter(self._train_dataloader))
        assert isinstance(img_batch, torch.Tensor)
        assert isinstance(label_batch, torch.Tensor)
        print(f"{img_batch.shape=} {label_batch.shape=}")
        
        assert img_batch.shape == (train_batch_size, 3, size, size)
        assert label_batch.shape == (train_batch_size,)
        
        self.example_input_array = torch.randn_like(img_batch)

        # self._model = ViTForImageClassification.from_pretrained(
        #     pretrained_name,
        #     num_labels=len(self.dataset.label_map),
        #     id2label=self.dataset.label_map,
        #     label2id=label2id)

        self._model = ResNetForImageClassification.from_pretrained(
            pretrained_name,
            num_labels=len(self.dataset.label_map),
            id2label=self.dataset.label_map,
            label2id=label2id,
            ignore_mismatched_sizes=True)

        assert isinstance(self._model, nn.Module)

        self.train_metrics: Optional[Metrics] = None
        self.val_metrics: Optional[Metrics] = None

    @property
    def num_classes(self):
        return len(self.dataset.label_map)
    
    @property
    def labelmap(self):
        return self.dataset.label_map

    def forward(self, img_batch):
        outputs = self._model(img_batch) # type: ignore
        return outputs.logits
        
    def common_step(self, batch, batch_idx):
        img_batch, label_batch = batch

        logits = self(img_batch)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, label_batch)
        preds_batch = logits.argmax(-1)

        return loss, preds_batch, label_batch

    def on_train_epoch_start(self) -> None:
        self.train_metrics = Metrics(
            self.num_classes,
            self.labelmap,
            "train",
            self.log).to(self.device)

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self.common_step(batch, batch_idx)
        assert self.train_metrics is not None
        self.train_metrics.update(loss, preds, labels)

        if False and batch_idx == 0:
            self._dump_train_images()

        return loss

    def _dump_train_images(self) -> None:
        img_batch, label_batch = next(iter(self._train_dataloader))
        for i_img, (img, label) in enumerate(zip(img_batch, label_batch)):
            img_np = img.cpu().numpy()
            denorm_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            img_uint8 = (255 * denorm_np).astype(np.uint8)
            pil_img = Image.fromarray(np.transpose(img_uint8, (1, 2, 0)))
            if self.logger is not None and self.logger.log_dir is not None:
                assert isinstance(self.logger.log_dir, str)
                os.makedirs(self.logger.log_dir, exist_ok=True)
                path = os.path.join(self.logger.log_dir,
                                    f"img_{i_img:02d}_{label.item()}.png")
                pil_img.save(path)

    def on_train_epoch_end(self) -> None:
        assert self.train_metrics is not None
        self.train_metrics.log()
        assert self.logger is not None
        if self.logger.log_dir is not None:
            path = os.path.join(self.logger.log_dir, "inference")
            self.save_checkpoint_dk(path)
    
    def save_checkpoint_dk(self, dirpath: str) -> None:
        if self.global_rank == 0:
            self._model.save_pretrained(dirpath)

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.common_step(batch, batch_idx)
        assert self.val_metrics is not None
        self.val_metrics.update(loss, preds, labels)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_metrics = Metrics(
            self.num_classes,
            self.labelmap,
            "val",
            self.log).to(self.device)
    
    def on_validation_epoch_end(self) -> None:
        assert self.val_metrics is not None
        self.val_metrics.log()

    def configure_optimizers(self):
        # No WD is the same as 1e-3 and better than 1e-2
        # LR 1e-3 is worse than 1e-4 (without LR scheduler)
        return AdamW(self.parameters(),
                     lr=1e-4,
                     )


def main():

    parser = ArgumentParser(description='KAUST-SDAIA Diabetic Retinopathy')
    parser.add_argument('--tag', action='store', type=str,
                        help='Extra suffix to put on the artefact dir name')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--convert-checkpoint', action='store', type=str,
                        help='Convert a checkpoint from training to pickle-independent '
                             'predictor-compatible directory')

    args = parser.parse_args()


    torch.set_float32_matmul_precision('high') # for V100/A100

    if args.convert_checkpoint is not None:

        print("Converting checkpoint", args.convert_checkpoint)

        checkpoint = torch.load(args.convert_checkpoint, map_location="cpu")
        print(list(checkpoint.keys()))

        model = ViTLightningModule.load_from_checkpoint(
            args.convert_checkpoint,
            map_location="cpu",
            hparams_file="tmp_ckpt_deleteme.yaml")

        model.save_checkpoint_dk("tmp_checkp_path_deleteme")

        print("Saved checkpoint. Done.")

    else:

        print("Start training")

        fast_dev_run = True if args.debug == True else False

        model = ViTLightningModule(fast_dev_run)

        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        art_dir_name = (f"{datetime_str}" +
                        (f"_{args.tag}" if args.tag is not None else ""))
        logger = TensorBoardLogger(save_dir=".", name="lightning_logs", version=art_dir_name)

        trainer = Trainer(
            logger=logger,
            benchmark=True,
            devices="auto",
            accelerator="auto",
            max_epochs=-1,
            callbacks=[
                ModelSummary(max_depth=-1),
                ],
            fast_dev_run=fast_dev_run,
            log_every_n_steps=10,
            )

        trainer.fit(
            model,
            train_dataloaders=model._train_dataloader,
            val_dataloaders=model._val_dataloader,
            )

        print("Training done")


if __name__ == "__main__":
    main()
