import io
from dataclasses import dataclass
from functools import reduce
from operator import add
from pathlib import Path
from typing import Collection, Optional, Sequence, TextIO, Union

import pandas as pd

from mmrr.utils.annotation import SentenceAnnotation
from mmrr.utils.prediction import SentencePrediction

from ..task import Task
from .mm_coreference import MultiModalCoreferenceResolutionEvaluator
from .mm_pas import MultiModalPASAnalysisEvaluator
from .utils import F1Metric


class MMRefEvaluator:
    """A data class to evaluate multi-modal reference resolution

    Args:
    tasks: 評価の対象とするタスク
    pas_cases: 評価対象の格
    """

    def __init__(
        self,
        tasks: Union[Collection[Task], Collection[str]],
        pas_cases: Collection[str],
    ) -> None:
        self.tasks: list[Task] = list(map(Task, tasks))
        self.pas_evaluator = MultiModalPASAnalysisEvaluator(pas_cases)
        self.coreference_evaluator = MultiModalCoreferenceResolutionEvaluator()

    def run(
        self,
        predicted_annotations: Sequence[SentencePrediction],
        gold_annotations: Sequence[SentenceAnnotation],
    ) -> "MMRefScore":
        assert len(predicted_annotations) == len(gold_annotations)
        results = []
        for predicted_annotation, gold_annotation in zip(
            predicted_annotations, gold_annotations
        ):
            assert predicted_annotation.sid == gold_annotation.sid
            results.append(self.run_single(predicted_annotation, gold_annotation))
        return reduce(add, results)

    def run_single(
        self,
        predicted_annotation: SentencePrediction,
        gold_annotation: SentenceAnnotation,
    ) -> "MMRefScore":
        if Task.MM_PAS_ANALYSIS in self.tasks:
            assert len(predicted_annotation.phrases) == len(gold_annotation.phrases)
            assert isinstance(gold_annotation.sid, str)
            pas_metrics = self.pas_evaluator.run(
                sid=gold_annotation.sid,
                predicted_phrases=predicted_annotation.phrases,
                gold_phrases=gold_annotation.phrases,
            )
        else:
            pas_metrics = None

        if Task.MM_COREFERENCE_RESOLUTION in self.tasks:
            assert len(predicted_annotation.phrases) == len(gold_annotation.phrases)
            assert isinstance(gold_annotation.sid, str)
            coreference_metrics = self.coreference_evaluator.run(
                sid=gold_annotation.sid,
                predicted_phrases=predicted_annotation.phrases,
                gold_phrases=gold_annotation.phrases,
            )
        else:
            coreference_metrics = None

        return MMRefScore(pas_metrics, coreference_metrics)


@dataclass(frozen=True)
class MMRefScore:
    """A data class for storing the numerical result of an evaluation"""

    pas_metrics: Optional[pd.DataFrame]
    coreference_metrics: Optional[pd.DataFrame]

    def to_dict(self) -> dict[str, dict[str, F1Metric]]:
        df_all = pd.DataFrame()
        if self.pas_metrics is not None:
            df_pas: pd.DataFrame = self.pas_metrics.copy()
            df_all = pd.concat([df_pas, df_all])
            df_all.loc["mm_pas"] = df_pas.sum(axis=0)
        if self.coreference_metrics is not None:
            df_coref = self.coreference_metrics.copy()
            df_all = pd.concat([df_all, df_coref])
            df_all.loc["mm_coreference"] = df_coref.sum(axis=0)

        return {
            k1: {k2: v2 for k2, v2 in v1.items() if pd.notna(v2)}
            for k1, v1 in df_all.to_dict(orient="index").items()
        }

    def export_txt(self, destination: Union[str, Path, TextIO]) -> None:
        """Export the evalutation results in a text format.

        Args:
            destination (Union[str, Path, TextIO]): 書き出す先
        """
        lines = []
        for rel_type, analysis_type_to_metric in self.to_dict().items():
            lines.append(rel_type)
            for analysis_type, metric in analysis_type_to_metric.items():
                lines.append(f"  {analysis_type}")
                lines.append(
                    f"    recall   : {metric.recall:.4f} ({metric.tp}/{metric.tp_fn})"
                )
        text = "\n".join(lines) + "\n"

        if isinstance(destination, (Path, str)):
            Path(destination).write_text(text)
        elif isinstance(destination, io.TextIOBase):
            destination.write(text)

    def export_csv(self, destination: Union[str, Path, TextIO], sep: str = ",") -> None:
        """Export the evaluation results in a csv format.

        Args:
            destination: 書き出す先
            sep: 区切り文字 (default: ',')
        """
        result_dict = self.to_dict()
        text = "task" + sep
        columns: list[str] = list(result_dict["pas"].keys())
        text += sep.join(columns) + "\n"
        for task, measures in result_dict.items():
            text += task + sep
            text += sep.join(
                f"{measures[column].f1:.6}" if column in measures else ""
                for column in columns
            )
            text += "\n"

        if isinstance(destination, (Path, str)):
            Path(destination).write_text(text)
        elif isinstance(destination, io.TextIOBase):
            destination.write(text)

    def __add__(self, other: "MMRefScore") -> "MMRefScore":
        if self.pas_metrics is not None:
            assert other.pas_metrics is not None
            pas_metrics = self.pas_metrics + other.pas_metrics
        else:
            pas_metrics = None
        if self.coreference_metrics is not None:
            assert other.coreference_metrics is not None
            coreference_metrics = self.coreference_metrics + other.coreference_metrics
        else:
            coreference_metrics = None
        return MMRefScore(pas_metrics, coreference_metrics)
