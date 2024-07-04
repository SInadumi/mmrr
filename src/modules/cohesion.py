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

from metrics import CohesionMetric
from modules.base import BaseModule
from modules.model.loss import binary_cross_entropy_with_logits, cross_entropy_loss
from utils.util import IGNORE_INDEX


class CohesionModule(BaseModule[CohesionMetric]):
    def __init__(self, hparams: DictConfig) -> None:
        analysis_target_threshold: float = getattr(
            hparams, "analysis_target_threshold", 0.5
        )  # default: 0.5
        super().__init__(hparams, CohesionMetric(analysis_target_threshold))

        self.model: nn.Module = hydra.utils.instantiate(
            hparams.model,
            num_relations=int("pas" in hparams.tasks) * len(hparams.cases)
            + int("coreference" in hparams.tasks)
            + int("bridging" in hparams.tasks),
        )

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        relation_logits, source_mask_logits = self.model(**batch)
        return {
            "relation_logits": relation_logits.masked_fill(
                ~batch["target_mask"], -1024.0
            ),
            "source_mask_logits": source_mask_logits,
        }

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        ret: dict[str, torch.Tensor] = self(batch)
        losses: dict[str, torch.Tensor] = {}

        source_mask: torch.Tensor = batch["source_mask"]  # (b, seq)
        target_mask: torch.Tensor = batch["target_mask"]  # (b, rel, seq, seq)

        relation_mask = (
            source_mask.unsqueeze(1).unsqueeze(3) & target_mask
        )  # (b, rel, seq, seq)
        losses["relation_loss"] = cross_entropy_loss(
            ret["relation_logits"], batch["target_label"], relation_mask
        )

        source_label: torch.Tensor = batch["source_label"]  # (b, task, seq)
        analysis_target_mask = source_label.ne(IGNORE_INDEX) & source_mask.unsqueeze(
            1
        )  # (b, task, seq)
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
        self, batch: dict, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        prediction = self.predict_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        metric = self.valid_corpus2metric[self.valid_corpora[dataloader_idx]]
        metric.update(prediction)

    @override
    def on_validation_epoch_end(self) -> None:
        metrics_log: dict[str, dict[str, float]] = {}
        for corpus, metric in self.valid_corpus2metric.items():
            assert isinstance(self.trainer.val_dataloaders, dict)
            metric.set_properties(
                {
                    "dataset": self.trainer.val_dataloaders[corpus].dataset,
                    "flip_writer_reader_according_to_type_id": self.hparams[
                        "flip_reader_writer"
                    ],
                }
            )
            metrics = metric.compute()
            metrics_log[corpus] = metrics
            metric.reset()

        for corpus, metrics in metrics_log.items():
            self.log_dict(
                {f"valid_{corpus}/{key}": value for key, value in metrics.items()}
            )
        for key in reduce(
            set.union, [set(metrics.keys()) for metrics in metrics_log.values()]
        ):
            mean_score = mean(
                metrics_log[corpus][key]
                for corpus in self.valid_corpora
                if key in metrics_log[corpus]
            )
            self.log(f"valid/{key}", mean_score)
        for key in reduce(
            set.union, [set(metrics.keys()) for metrics in metrics_log.values()]
        ):
            mean_score = mean(
                metrics_log[corpus][key]
                for corpus in self.valid_corpora
                if key in metrics_log[corpus] and corpus != "fuman"
            )
            self.log(f"valid_wo_fuman/{key}", mean_score)

    @rank_zero_only
    def on_train_end(self) -> None:
        best_model_path: str = self.trainer.checkpoint_callback.best_model_path  # type: ignore
        if not best_model_path:
            return
        save_dir = Path(self.hparams.exp_dir) / self.hparams.run_id  # type: ignore
        best_path = save_dir / "best.ckpt"
        if best_path.exists():
            best_path.unlink()
        actual_best_path = Path(best_model_path)
        assert actual_best_path.parent.resolve() == best_path.parent.resolve()
        best_path.resolve().symlink_to(actual_best_path.name)

    @override
    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        prediction = self.predict_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        metric = self.test_corpus2metric[self.test_corpora[dataloader_idx]]
        metric.update(prediction)

    @override
    def on_test_epoch_end(self) -> None:
        metrics_log = {}
        for corpus, metric in self.test_corpus2metric.items():
            assert isinstance(self.trainer.test_dataloaders, dict)
            metric.set_properties(
                {
                    "dataset": self.trainer.test_dataloaders[corpus].dataset,
                    "flip_writer_reader_according_to_type_id": self.hparams[
                        "flip_reader_writer"
                    ],
                }
            )
            metrics = metric.compute()
            metrics_log[corpus] = metrics
            metric.reset()

        for corpus, metrics in metrics_log.items():
            self.log_dict(
                {f"test_{corpus}/{key}": value for key, value in metrics.items()}
            )
        for key in reduce(
            set.union, [set(metrics.keys()) for metrics in metrics_log.values()]
        ):
            mean_score = mean(
                metrics_log[corpus][key]
                for corpus in self.test_corpora
                if key in metrics_log[corpus]
            )
            self.log(f"test/{key}", mean_score)
        for key in reduce(
            set.union, [set(metrics.keys()) for metrics in metrics_log.values()]
        ):
            mean_score = mean(
                metrics_log[corpus][key]
                for corpus in self.test_corpora
                if key in metrics_log[corpus] and corpus != "fuman"
            )
            self.log(f"test_wo_fuman/{key}", mean_score)

    @override
    def predict_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Any]:
        output: dict[str, torch.Tensor] = self(batch)
        return {
            "example_ids": batch["example_id"],
            "dataloader_idx": dataloader_idx,
            **output,
        }
