from typing import Collection

import pandas as pd

from utils.annotation import PhraseAnnotation
from utils.prediction import BoundingBoxPrediction, PhrasePrediction

from .utils import RECALL_TOP_KS, F1Metric


class VisPASAnalysisEvaluator:
    """A class to evaluate visually-grounded predicate argument structure analysis"""

    def __init__(
        self,
        cases: Collection[str],
    ) -> None:
        self.comp_result: dict[tuple, str] = {}
        # NOTE: 現在の実装では非総称名詞は評価の対象外 (e.g. "ガ≒", "ヲ≒", "ニ≒", ...)
        self.cases: list[str] = list(cases)
        self.analysis: dict[int, str] = {k: f"recall@{k}" for k in RECALL_TOP_KS}

    def run(
        self,
        sid: str,
        predicted_phrases: PhrasePrediction,
        gold_phrases: PhraseAnnotation,
    ) -> pd.DataFrame:
        assert len(predicted_phrases) == len(gold_phrases)
        metrics = pd.DataFrame(
            [[F1Metric() for _ in self.analysis.values()] for _ in self.cases],
            index=self.cases,
            columns=list(self.analysis.values()),
        )

        local_comp_result: dict[tuple, str] = {}
        for idx, (predicted_mention, gold_mention) in enumerate(
            zip(predicted_phrases, gold_phrases)
        ):
            # NOTE: `is_target==False` ならば candidatesは空集合
            candidates: dict[str, list[BoundingBoxPrediction]] = {}
            for rel_pred in predicted_mention.relations:
                candidates[rel_pred.type] = rel_pred.bounding_box

            for pas_case in self.cases:
                _gold_case_relations = [
                    rel for rel in gold_mention.relations if rel.type == pas_case
                ]
                if len(_gold_case_relations) == 0:
                    continue
                key = (f"{idx}:{predicted_mention.text}", pas_case)

                # Compute recall
                # 正解が複数ある場合、そのうち一つが当てられていればそれを正解に採用
                for recall_top_k, _metric_name in self.analysis.items():
                    local_comp_result[key] = f"{pas_case}:{_metric_name}"
                    metrics.loc[pas_case, _metric_name].tp_fn += 1
                    for rel in _gold_case_relations:

                        _topk = recall_top_k
                        if pas_case not in candidates or len(candidates[pas_case]) == 0:
                            break
                        elif len(candidates[pas_case]) < recall_top_k:
                            _topk = len(candidates[pas_case])

                        if rel.classId in set(
                            int(c.class_id) for c in candidates[pas_case][:_topk]
                        ):
                            metrics.loc[pas_case, _metric_name].tp += 1
                            break
        self.comp_result.update({(sid, *k): v for k, v in local_comp_result.items()})
        return metrics
