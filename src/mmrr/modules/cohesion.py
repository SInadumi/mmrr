from functools import reduce
from statistics import mean
from typing import Optional, Union

import hydra
import torch
from omegaconf import DictConfig
from transformers import AutoModel, PreTrainedModel
from typing_extensions import override

from mmrr.metrics import CohesionMetric
from mmrr.modules.model.loss import (
    ContrastiveLoss,
    SupConLoss,
    binary_cross_entropy_with_logits,
    cross_entropy_loss,
)
from mmrr.utils.util import IGNORE_INDEX

from .base import BaseModule

LossType = Union[ContrastiveLoss, SupConLoss]


class CohesionModule(BaseModule[CohesionMetric]):
    def __init__(self, hparams: DictConfig) -> None:
        analysis_target_threshold: float = getattr(
            hparams, "analysis_target_threshold", 0.5
        )  # default: 0.5
        super().__init__(hparams, CohesionMetric(analysis_target_threshold))
        self.encoder: PreTrainedModel = AutoModel.from_pretrained(
            pretrained_model_name_or_path=hparams.model_name_or_path
        )
        self.encoder.resize_token_embeddings(
            new_num_tokens=self.encoder.config.vocab_size
            + len(hparams.exophora_referents)
            + 2,  # +2: [NULL] and [NA]
        )
        num_relation_types = (
            int("pas" in hparams.tasks) * len(hparams.cases)
            + int("coreference" in hparams.tasks)
            + int("bridging" in hparams.tasks)
        )
        hidden_size = self.encoder.config.hidden_size
        self.relation_classifier = hydra.utils.instantiate(
            hparams.model.relation_head,
            num_relations=num_relation_types,
            hidden_size=hidden_size,
        )
        self.analysis_target_classifier = hydra.utils.instantiate(
            hparams.model.binary_head, hidden_size=hidden_size
        )

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        encoded = self.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
        ).last_hidden_state  # (b, seq) -> (b, seq, hid)
        relation_logits = self.relation_classifier(encoded)
        source_mask_logits = self.analysis_target_classifier(encoded)
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
