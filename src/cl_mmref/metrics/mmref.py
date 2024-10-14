from statistics import mean
from typing import Optional

import numpy as np
import torch
from rhoknp import Sentence

from cl_mmref.cohesion_tools.constants import RECALL_TOP_KS
from cl_mmref.cohesion_tools.evaluators.mmref import MMRefEvaluator, MMRefScore
from cl_mmref.cohesion_tools.evaluators.utils import F1Metric
from cl_mmref.cohesion_tools.task import Task
from cl_mmref.datamodule.example import MMRefExample
from cl_mmref.datasets.mmref_dataset import MMRefDataset
from cl_mmref.utils.annotation import SentenceAnnotation
from cl_mmref.utils.prediction import SentencePrediction
from cl_mmref.writer.mmref import SentenceJsonWriter

from .base import BaseModuleMetric


class MMRefMetric(BaseModuleMetric):
    full_state_update: bool = False
    STATE_NAMES = (
        "example_ids",
        "relation_logits",
        "source_mask_logits",
    )

    def __init__(self, analysis_target_threshold: float = None) -> None:
        super().__init__()
        self.analysis_target_threshold = analysis_target_threshold
        self.dataset: Optional[MMRefDataset] = None
        self.example_ids: torch.Tensor  # (N)
        self.relation_logits: torch.Tensor  # (N, rel, t_seq, v_seq)
        self.source_mask_logits: torch.Tensor  # (N, task, t_seq)

    def compute(self) -> dict[str, float]:
        for state_name in self.STATE_NAMES:
            state = getattr(self, state_name)
            if not isinstance(state, torch.Tensor):
                setattr(self, state_name, torch.cat(state, dim=0))

        predicted_annotations, gold_annotations = self._build_annotations()

        metrics: dict[str, float] = {}
        assert self.dataset is not None, "dataset is not set"
        metrics.update(
            self._compute_analysis_target_metrics(
                dataset=self.dataset,
                predicted_annotations=predicted_annotations,
                gold_annotations=gold_annotations,
            )
        )
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
        json_writer = SentenceJsonWriter(self.dataset)
        assert (
            len(self.example_ids)
            == len(self.relation_logits)
            == len(self.source_mask_logits)
        )
        for example_id, relation_logits, source_mask_logits in zip(
            self.example_ids, self.relation_logits, self.source_mask_logits
        ):
            gold_example: MMRefExample = self.dataset.examples[example_id.item()]
            gold_sentences: list[SentenceAnnotation] = [
                self.dataset.sid2vis_sentence[sid]
                for sid in gold_example.sentence_indices
            ]

            # (phrase, rel, candidate)
            relation_prediction: np.ndarray = self.dataset.dump_relation_prediction(
                relation_logits.cpu().numpy(), gold_example
            )
            # (phrase, rel, candidate)
            candidate_selection_prediction: np.ndarray = np.argsort(
                -relation_prediction, axis=2
            )  # descending order
            # (phrase, task)
            source_mask_prediction: np.ndarray = (
                self.dataset.dump_source_mask_prediction(
                    source_mask_logits.cpu().numpy(), gold_example
                )
            )
            assert self.analysis_target_threshold is not None
            is_analysis_target: np.ndarray = (
                source_mask_prediction >= self.analysis_target_threshold
            )  # (phrase, task)

            sentence_predictions.extend(
                json_writer.write_sentence_predictions(
                    gold_example,
                    gold_sentences,
                    candidate_selection_prediction.tolist(),
                    is_analysis_target.tolist(),
                )
            )
            sentence_annotations.extend(gold_sentences)

        return sentence_predictions, sentence_annotations

    @staticmethod
    def _compute_analysis_target_metrics(
        dataset: MMRefDataset,
        predicted_annotations: list[SentencePrediction],
        gold_annotations: list[SentenceAnnotation],
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        task_to_analysis_target_metric: dict[Task, F1Metric] = {
            task: F1Metric() for task in dataset.tasks
        }
        for predicted_annotation, gold_annotation in zip(
            predicted_annotations, gold_annotations
        ):
            assert predicted_annotation.sid == gold_annotation.sid

            for task in dataset.tasks:
                metric = task_to_analysis_target_metric[task]
                extractor = dataset.task_to_extractor[task]
                predicted_phrases = [
                    ph for ph in predicted_annotation.phrases if ph.task == task
                ]
                assert len(predicted_phrases) == len(gold_annotation.phrases)

                for predicted_phrase, gold_phrase in zip(
                    predicted_phrases, gold_annotation.phrases
                ):
                    predicted_is_target: bool = len(predicted_phrase.relations) > 0
                    gold_is_target: bool = extractor.is_target(gold_phrase)
                    if predicted_is_target is True:
                        if gold_is_target is True:
                            metric.tp += 1
                        metric.tp_fp += 1
                    if gold_is_target is True:
                        if predicted_is_target is True:
                            pass
                        metric.tp_fn += 1
        for task, metric in task_to_analysis_target_metric.items():
            for metric_name in ("f1", "precision", "recall"):
                metrics[f"{task.value}_target_{metric_name}"] = getattr(
                    metric, metric_name
                )
        for metric_name in ("f1", "precision", "recall"):
            metrics[f"mmref_target_{metric_name}"] = mean(
                getattr(metric, metric_name)
                for metric in task_to_analysis_target_metric.values()
            )
        return metrics

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
                for key in (f"vis_pas_{_metric_name}", f"vis_coref_{_metric_name}")
                if key in metrics
            )
        return metrics
