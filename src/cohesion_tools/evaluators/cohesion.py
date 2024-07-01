import io
from dataclasses import dataclass
from functools import reduce
from operator import add
from pathlib import Path
from typing import Collection, Dict, List, Optional, Sequence, TextIO, Union

import pandas as pd
from rhoknp import Document
from rhoknp.cohesion import (
    ExophoraReferentType,
)

from cohesion_tools.evaluators.bridging import BridgingReferenceResolutionEvaluator
from cohesion_tools.evaluators.coreference import CoreferenceResolutionEvaluator
from cohesion_tools.evaluators.pas import PASAnalysisEvaluator
from cohesion_tools.evaluators.utils import F1Metric
from cohesion_tools.task import Task


class CohesionEvaluator:
    """結束性解析の評価を行うクラス

    To evaluate system output with this class, you have to prepare gold data and system prediction data as instances of
    :class:`rhoknp.Document`

    Args:
        tasks: 評価の対象とするタスク (cohesion_tools.task.Task を参照)
        exophora_referent_types: 評価の対象とする外界照応の照応先 (rhoknp.cohesion.ExophoraReferentTypeType を参照)
        pas_cases: 述語項構造の評価の対象とする格 (rhoknp.cohesion.rel.CASE_TYPES を参照)
        bridging_rel_types: 橋渡し参照解析の評価の対象とする関係 (rhoknp.cohesion.rel.CASE_TYPES を参照．default: ["ノ"])
    """

    def __init__(
        self,
        tasks: Union[Collection[Task], Collection[str]],
        exophora_referent_types: Collection[ExophoraReferentType],
        pas_cases: Collection[str],
        bridging_rel_types: Optional[Collection[str]] = None,
    ) -> None:
        self.exophora_referent_types: List[ExophoraReferentType] = list(exophora_referent_types)
        self.pas_cases: List[str] = list(pas_cases)
        self.tasks: List[Task] = list(map(Task, tasks))
        self.pas_evaluator = PASAnalysisEvaluator(exophora_referent_types, pas_cases)
        self.bridging_evaluator = BridgingReferenceResolutionEvaluator(
            exophora_referent_types,
            bridging_rel_types if bridging_rel_types is not None else ["ノ"],
        )
        self.coreference_evaluator = CoreferenceResolutionEvaluator(exophora_referent_types)

    def run(self, predicted_documents: Sequence[Document], gold_documents: Sequence[Document]) -> "CohesionScore":
        """読み込んだ正解文書集合とシステム予測文書集合に対して評価を行う

        Args:
            predicted_documents: システム予測文書集合
            gold_documents: 正解文書集合

        Returns:
            CohesionScore: 評価結果のスコア
        """
        # long document may have been ignored
        assert {d.doc_id for d in predicted_documents} <= {d.doc_id for d in gold_documents}
        doc_ids: List[str] = [d.doc_id for d in predicted_documents]
        doc_id2predicted_document: Dict[str, Document] = {d.doc_id: d for d in predicted_documents}
        doc_id2gold_document: Dict[str, Document] = {d.doc_id: d for d in gold_documents}

        results = []
        for doc_id in doc_ids:
            predicted_document = doc_id2predicted_document[doc_id]
            gold_document = doc_id2gold_document[doc_id]
            results.append(self.run_single(predicted_document, gold_document))
        return reduce(add, results)

    def run_single(self, predicted_document: Document, gold_document: Document) -> "CohesionScore":
        """Compute cohesion scores for a pair of gold document and predicted document"""
        assert len(predicted_document.base_phrases) == len(gold_document.base_phrases)

        pas_metrics = (
            self.pas_evaluator.run(predicted_document, gold_document) if Task.PAS_ANALYSIS in self.tasks else None
        )
        bridging_metrics = (
            self.bridging_evaluator.run(predicted_document, gold_document)
            if Task.BRIDGING_REFERENCE_RESOLUTION in self.tasks
            else None
        )
        coreference_metric = (
            self.coreference_evaluator.run(predicted_document, gold_document)
            if Task.COREFERENCE_RESOLUTION in self.tasks
            else None
        )

        return CohesionScore(pas_metrics, bridging_metrics, coreference_metric)


@dataclass(frozen=True)
class CohesionScore:
    """A data class for storing the numerical result of an evaluation"""

    pas_metrics: Optional[pd.DataFrame]
    bridging_metrics: Optional[pd.DataFrame]
    coreference_metrics: Optional[pd.Series]

    def to_dict(self) -> Dict[str, Dict[str, F1Metric]]:
        """Convert data to dictionary"""
        df_all = pd.DataFrame(index=["pas"])
        if self.pas_metrics is not None:
            df_pas: pd.DataFrame = self.pas_metrics.copy()
            df_pas["overt_dep"] = df_pas["overt"] + df_pas["dep"]
            df_pas["endophora"] = df_pas["overt"] + df_pas["dep"] + df_pas["zero_endophora"]
            df_pas["zero"] = df_pas["zero_endophora"] + df_pas["exophora"]
            df_pas["dep_zero"] = df_pas["dep"] + df_pas["zero"]
            df_pas["all"] = df_pas["overt"] + df_pas["dep_zero"]
            df_pas = df_pas.rename(index=lambda x: f"pas_{x}")
            df_all = pd.concat([df_pas, df_all])
            df_all.loc["pas"] = df_pas.sum(axis=0)

        if self.bridging_metrics is not None:
            df_bridging: pd.DataFrame = self.bridging_metrics.copy()
            df_bridging["endophora"] = df_bridging["dep"] + df_bridging["zero_endophora"]
            df_bridging["zero"] = df_bridging["zero_endophora"] + df_bridging["exophora"]
            df_bridging["dep_zero"] = df_bridging["dep"] + df_bridging["zero"]
            df_bridging["all"] = df_bridging["dep_zero"]
            df_bridging = df_bridging.rename(index=lambda x: f"bridging_{x}")
            df_all = pd.concat([df_all, df_bridging])
            df_all.loc["bridging"] = df_bridging.sum(axis=0)

        if self.coreference_metrics is not None:
            df_coref = self.coreference_metrics.copy()
            df_coref["all"] = df_coref["endophora"] + df_coref["exophora"]
            df_all.loc["coreference"] = df_coref

        return {
            k1: {k2: v2 for k2, v2 in v1.items() if pd.notna(v2)} for k1, v1 in df_all.to_dict(orient="index").items()
        }

    def export_txt(self, destination: Union[str, Path, TextIO]) -> None:
        """Export the evaluation results in a text format.

        Args:
            destination (Union[str, Path, TextIO]): 書き出す先
        """
        lines = []
        for rel_type, analysis_type_to_metric in self.to_dict().items():
            lines.append(rel_type)
            for analysis_type, metric in analysis_type_to_metric.items():
                lines.append(f"  {analysis_type}")
                lines.append(f"    precision: {metric.precision:.4f} ({metric.tp}/{metric.tp_fp})")
                lines.append(f"    recall   : {metric.recall:.4f} ({metric.tp}/{metric.tp_fn})")
                lines.append(f"    F        : {metric.f1:.4f}")
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
        columns: List[str] = list(result_dict["pas"].keys())
        text += sep.join(columns) + "\n"
        for task, measures in result_dict.items():
            text += task + sep
            text += sep.join(f"{measures[column].f1:.6}" if column in measures else "" for column in columns)
            text += "\n"

        if isinstance(destination, (Path, str)):
            Path(destination).write_text(text)
        elif isinstance(destination, io.TextIOBase):
            destination.write(text)

    def __add__(self, other: "CohesionScore") -> "CohesionScore":
        if self.pas_metrics is not None:
            assert other.pas_metrics is not None
            pas_metrics = self.pas_metrics + other.pas_metrics
        else:
            pas_metrics = None
        if self.bridging_metrics is not None:
            assert other.bridging_metrics is not None
            bridging_metrics = self.bridging_metrics + other.bridging_metrics
        else:
            bridging_metrics = None
        if self.coreference_metrics is not None:
            assert other.coreference_metrics is not None
            coreference_metric = self.coreference_metrics + other.coreference_metrics
        else:
            coreference_metric = None
        return CohesionScore(pas_metrics, bridging_metrics, coreference_metric)
