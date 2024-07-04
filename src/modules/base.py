from copy import deepcopy
from typing import Generic, TypeVar

import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig

from metrics.base import BaseModuleMetric
from utils.util import oc_resolve

MetricType = TypeVar("MetricType", bound=BaseModuleMetric)


class BaseModule(pl.LightningModule, Generic[MetricType]):
    def __init__(self, hparams: DictConfig, metric: MetricType) -> None:
        super().__init__()
        oc_resolve(hparams, keys=hparams.keys_to_resolve)
        # this line allows accessing init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(hparams)
        if valid_corpora := getattr(hparams.datamodule, "valid", None):
            self.valid_corpora: list[str] = list(valid_corpora)
            self.valid_corpus2metric: dict[str, MetricType] = {
                corpus: deepcopy(metric) for corpus in valid_corpora
            }
        if test_corpora := getattr(hparams.datamodule, "test", None):
            self.test_corpora: list[str] = list(test_corpora)
            self.test_corpus2metric: dict[str, MetricType] = {
                corpus: deepcopy(metric) for corpus in test_corpora
            }

    def configure_optimizers(self):
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ("bias", "LayerNorm.weight")
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.hparams.optimizer.weight_decay,
                "name": "decay",
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
                "name": "no_decay",
            },
        ]
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer,
            params=optimizer_grouped_parameters,
            _convert_="partial",
        )
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = (
            self.hparams.warmup_steps or total_steps * self.hparams.warmup_ratio
        )
        if hasattr(self.hparams.scheduler, "num_warmup_steps"):
            self.hparams.scheduler.num_warmup_steps = warmup_steps
        if hasattr(self.hparams.scheduler, "num_training_steps"):
            self.hparams.scheduler.num_training_steps = total_steps
        lr_scheduler = hydra.utils.instantiate(
            self.hparams.scheduler, optimizer=optimizer
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
