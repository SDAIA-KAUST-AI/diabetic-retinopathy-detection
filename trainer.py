import os
from typing import Optional
import numpy as np
import math
from PIL import Image

import torch
import torch.nn as nn
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

# from transformers import ViTImageProcessor
# from transformers import ViTForImageClassification
from transformers import AdamW
from transformers import AutoImageProcessor, ResNetForImageClassification
import lightning as L

from data import RetinopathyDataset, Split
from metrics import Metrics


def worker_init_fn(worker_id: int) -> None:
    """ Initialize workers in a way that they draw different
    random samples and do not repeat identical pseudorandom
    sequences of each other, which may be the case with Fork
    multiprocessing.

    Args:
        worker_id (int): id of a preprocessing worker process launched
        by one DDP training process. 
    """
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
    """ Lightning Module that implements neural network training hooks. """
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
        """ Save augmented images to disk for inspection. """
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
