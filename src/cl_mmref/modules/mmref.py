from functools import reduce
from statistics import mean
from typing import Optional

import hydra
import torch
from omegaconf import DictConfig
from transformers import AutoModel, PreTrainedModel
from typing_extensions import override

from cl_mmref.metrics import MMRefMetric
from cl_mmref.modules.model.loss import cross_entropy_loss

from .base import BaseModule

IGNORE_INDEX = -100


class MMRefModule(BaseModule[MMRefMetric]):
    def __init__(self, hparams: DictConfig):
        super().__init__(hparams, MMRefMetric())
        self.encoder: PreTrainedModel = AutoModel.from_pretrained(
            pretrained_model_name_or_path=hparams.model_name_or_path
        )
        self.encoder.resize_token_embeddings(
            new_num_tokens=self.encoder.config.vocab_size
            + len(hparams.exophora_referents)
            + 2,  # +2: [NULL] and [NA]
        )
        num_relation_types = int("mm_pas" in hparams.tasks) * len(hparams.cases) + int(
            "mm_coreference" in hparams.tasks
        )
        hidden_size = self.encoder.config.hidden_size
        self.relation_classifier = hydra.utils.instantiate(
            hparams.model.relation_head,
            num_relations=num_relation_types,
            source_hidden_size=hidden_size,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        encoded_text = self.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
        ).last_hidden_state  # (b, seq) -> (b, seq, hid)
        relation_logits = self.relation_classifier(
            source_hidden_state=encoded_text, target_hidden_state=batch["vis_embeds"]
        )
        return {
            "relation_logits": relation_logits.masked_fill(
                ~batch["vis_attention_mask"].unsqueeze(1).unsqueeze(2), -1024.0
            ),
        }

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        ret: dict[str, torch.Tensor] = self(batch)
        losses: dict[str, torch.Tensor] = {}
        relation_mask: torch.Tensor = batch["target_mask"]  # (b, rel, seq, seq)
        losses["relation_loss"] = cross_entropy_loss(
            ret["relation_logits"], batch["target_label"], relation_mask
        )

        losses["loss"] = losses["relation_loss"]
        self.log_dict({f"train/{key}": value for key, value in losses.items()})
        return losses["loss"]

    @override
    def on_validation_epoch_end(self) -> None:
        metrics_log: dict[str, dict[str, float]] = {}
        for corpus, metric in self.valid_corpus2metric.items():
            assert isinstance(self.trainer.val_dataloaders, dict)
            metric.set_properties(
                {"dataset": self.trainer.val_dataloaders[corpus].dataset}
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

    @override
    def on_test_epoch_end(self) -> None:
        metrics_log: dict[str, dict[str, float]] = {}
        for corpus, metric in self.test_corpus2metric.items():
            assert isinstance(self.trainer.test_dataloaders, dict)
            metric.set_properties(
                {"dataset": self.trainer.test_dataloaders[corpus].dataset}
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
