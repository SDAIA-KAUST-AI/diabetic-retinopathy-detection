from typing import Dict, Callable

import torch

from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification.accuracy import MulticlassAccuracy
from torchmetrics.classification import MulticlassCohenKappa


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

