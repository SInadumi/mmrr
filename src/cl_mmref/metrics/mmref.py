from statistics import mean
from typing import Optional

import numpy as np
import torch

from cl_mmref.datamodule.example import MMRefExample
from cl_mmref.datasets.mmref_dataset import MMRefDataset
from cl_mmref.tools.constants import RECALL_TOP_KS
from cl_mmref.tools.evaluators.mmref import MMRefEvaluator, MMRefScore
from cl_mmref.utils.annotation import SentenceAnnotation
from cl_mmref.utils.prediction import SentencePrediction
from cl_mmref.writer.mmref import ProbabilityJsonWriter

from .base import BaseModuleMetric


class MMRefMetric(BaseModuleMetric):
    full_state_update: bool = False
    STATE_NAMES = (
        "example_ids",
        "relation_logits",
    )

    def __init__(self) -> None:
        super().__init__()
        self.dataset: Optional[MMRefDataset] = None
        self.example_ids: torch.Tensor  # (N)
        self.relation_logits: torch.Tensor  # (N, rel, seq, seq)

    def compute(self) -> dict[str, float]:
        for state_name in self.STATE_NAMES:
            state = getattr(self, state_name)
            if not isinstance(state, torch.Tensor):
                setattr(self, state_name, torch.cat(state, dim=0))

        predicted_annotations, gold_annotations = self._build_annotations()

        metrics: dict[str, float] = {}
        assert self.dataset is not None, "dataset is not set"
        metrics.update(
            self._compute_mmref_metrics(
                dataset=self.dataset,
                predicted_annotations=predicted_annotations,
                gold_annotations=gold_annotations,
            )
        )
        return metrics

    def _build_annotations(
        self,
    ) -> tuple[list[SentencePrediction], list[SentenceAnnotation]]:
        sentence_predictions: list[SentencePrediction] = []
        sentence_annotations: list[SentenceAnnotation] = []
        assert self.dataset is not None, "dataset is not set"
        json_writer = ProbabilityJsonWriter(self.dataset)
        assert len(self.example_ids) == len(self.relation_logits)
        for example_id, relation_logits in zip(self.example_ids, self.relation_logits):
            gold_example: MMRefExample = self.dataset.examples[example_id.item()]
            gold_sentences: list[SentenceAnnotation] = [
                self.dataset.sid2vis_sentence[sid]
                for sid in gold_example.sentence_indices
            ]

            # (phrase, rel, candidate)
            relation_prediction: np.ndarray = self.dataset.dump_relation_prediction(
                relation_logits.cpu().numpy(), gold_example
            )

            # descending order
            predicted_candidates: np.ndarray = np.argsort(-relation_prediction, axis=2)
            predicted_probabilities: np.ndarray = (-1) * np.sort(
                -relation_prediction, axis=2
            )
            assert predicted_candidates.size == predicted_probabilities.size

            sentence_predictions.extend(
                json_writer.write_sentence_predictions(
                    gold_example,
                    gold_sentences,
                    predicted_candidates.tolist(),
                    predicted_probabilities.tolist(),
                )
            )
            sentence_annotations.extend(gold_sentences)

        return sentence_predictions, sentence_annotations

    @staticmethod
    def _compute_mmref_metrics(
        dataset: MMRefDataset,
        predicted_annotations: list[SentencePrediction],
        gold_annotations: list[SentenceAnnotation],
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        evaluator = MMRefEvaluator(tasks=dataset.tasks, pas_cases=dataset.cases)
        score: MMRefScore = evaluator.run(
            gold_annotations=gold_annotations,
            predicted_annotations=predicted_annotations,
        )

        for task_str, analysis_type_to_metric in score.to_dict().items():
            for analysis_type, metric in analysis_type_to_metric.items():
                key = task_str
                if analysis_type != "all":
                    key += f"_{analysis_type}"
                # metrics[key + "_tp_fn"] = (
                #     metric.tp_fn
                # )  # FIXME: This is redundant variable.
                metrics[key] = metric.recall
        for recall_top_k in RECALL_TOP_KS:
            _metric_name = f"recall@{recall_top_k}"
            metrics[f"mmref_{_metric_name}"] = mean(
                metrics[key]
                for key in (f"mm_pas_{_metric_name}", f"mm_coreference_{_metric_name}")
                if key in metrics
            )
        return metrics
