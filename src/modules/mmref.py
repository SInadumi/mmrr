from functools import reduce
from pathlib import Path
from statistics import mean
from typing import Any, Optional

import hydra
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig
from torch import nn
from typing_extensions import override

from metrics.mmref import MMRefMetric
from modules.base import BaseModule
from modules.model.loss import binary_cross_entropy_with_logits, cross_entropy_loss

IGNORE_INDEX = -100


class MMRefModule(BaseModule[MMRefMetric]):
    def __init__(self, hparams: DictConfig):
        super().__init__(hparams, MMRefMetric)
        self.model: nn.Module = hydra.utils.instantiate(
            hparams.model,
            num_relations=int("vis_pas" in hparams.tasks) * len(hparams.cases)
            + int("vis_coreference" in hparams.tasks),
        )

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        relation_logits, source_mask_logits, h_src, h_tgt = self.model(**batch)
        return {
            "relation_logits": relation_logits.masked_fill(
                ~batch["target_mask"], -1024.0
            ),
            "source_mask_logits": source_mask_logits,
            "h_src": h_src,
            "h_tgt": h_tgt,
        }

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        ret: dict[str, torch.Tensor] = self(batch)
        losses: dict[str, torch.Tensor] = {}

        txt_source_mask: torch.Tensor = batch["source_mask"]  # (b, seq1)
        relation_mask: torch.Tensor = batch["target_mask"]  # (b, rel, seq1, seq2)

        losses["relation_loss"] = cross_entropy_loss(
            ret["relation_logits"], batch["target_label"], relation_mask
        )

        source_label: torch.Tensor = batch["source_label"]  # (b, task, seq1)
        analysis_target_mask = source_label.ne(IGNORE_INDEX) & txt_source_mask.unsqueeze(
            1
        )  # (b, task, seq1)
        source_label = torch.where(
            analysis_target_mask, source_label, torch.zeros_like(source_label)
        )
        losses["source_mask_loss"] = binary_cross_entropy_with_logits(
            ret["source_mask_logits"], source_label, analysis_target_mask
        )
        # weighted sum
        losses["loss"] = losses["relation_loss"] + losses["source_mask_loss"] * 0.5

        self.log_dict({f"train/{key}": value for key, value in losses.items()})
        return losses["loss"]

    @override
    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        pass

    @override
    def on_validation_epoch_end(self) -> None:
        pass

    @rank_zero_only
    def on_train_end(self) -> None:
        pass

    @override
    def test_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        pass

    @override
    def on_test_epoch_end(self) -> None:
        pass

    @override
    def predict_step(
        self, batch: dict[str, torch.Tensor], dataloader_idx: int = 0
    ) -> dict[str, Any]:
        pass
