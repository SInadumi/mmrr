from dataclasses import fields, is_dataclass
from typing import Any, Optional

import hydra
import lightning.pytorch as pl
import torch
from lightning.pytorch.trainer.states import TrainerFn
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import ConcatDataset


class MTDataModule(pl.LightningDataModule):
    # 0: Define Configure
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg: DictConfig = cfg
        self.batch_size: int = cfg.batch_size
        self.num_workers: int = cfg.num_workers
        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: dict[str, Dataset] = {}
        self.test_datasets: dict[str, Dataset] = {}
        self.predict_dataset: Optional[Dataset] = None

    # 1. Download Dataset
    def prepare_data(self):
        pass

    # 2. Load Dataset Partition (train, valid, test, predict)
    def setup(self, stage: Optional[str] = None):
        if stage == TrainerFn.FITTING:
            self.train_dataset = ConcatDataset(
                [hydra.utils.instantiate(conf) for conf in self.cfg.train.values()]
            )
        if stage in (TrainerFn.FITTING, TrainerFn.VALIDATING, TrainerFn.TESTING):
            self.val_datasets = {
                corpus: hydra.utils.instantiate(conf)
                for corpus, conf in self.cfg.valid.items()
            }
        if stage == TrainerFn.TESTING:
            self.test_datasets = {
                corpus: hydra.utils.instantiate(conf)
                for corpus, conf in self.cfg.test.items()
            }
        if stage == TrainerFn.PREDICTING:
            if self.cfg.predict.data_path is not None:
                self.predict_dataset = hydra.utils.instantiate(self.cfg.predict)

    # 3. Return Train Dataloader from "self.train_dataset"
    # NOTE: When running under a distributed strategy, Lightning handles the distributed sampler for you by default.
    # cf.) https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return self._get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> dict[str, DataLoader]:
        return {
            corpus: self._get_dataloader(dataset, shuffle=False)
            for corpus, dataset in self.val_datasets.items()
        }

    def test_dataloader(self) -> dict[str, DataLoader]:
        return {
            corpus: self._get_dataloader(dataset, shuffle=False)
            for corpus, dataset in self.test_datasets.items()
        }

    def predict_dataloader(self) -> DataLoader:
        assert self.predict_dataset is not None
        return self._get_dataloader(self.predict_dataset, shuffle=False)

    def _get_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=self._dataclass_data_collator,
        )

    @staticmethod
    def _dataclass_data_collator(features: list[Any]) -> dict[str, torch.Tensor]:
        first: Any = features[0]
        assert is_dataclass(first), "Data must be a dataclass"
        batch: dict[str, torch.Tensor] = {}
        for field in fields(first):
            feats = [getattr(f, field.name) for f in features]
            if type(feats[0]) == torch.Tensor:  # vis_input_embeds
                batch[field.name] = torch.stack(feats)
            else:
                batch[field.name] = torch.as_tensor(feats)
        return batch
