from typing import Collection

import pandas as pd

from utils.annotation import BoundingBox, PhraseAnnotation
from utils.prediction import BoundingBoxPrediction, PhrasePrediction

from .utils import IOU_THRESHOLD, RECALL_TOP_KS, F1Metric, box_iou


class VisPASAnalysisEvaluator:
    """A class to evaluate visually-grounded predicate argument structure analysis"""

    def __init__(
        self,
        cases: Collection[str],
    ) -> None:
        self.comp_result: dict[tuple, str] = {}
        # NOTE: 現在の実装では総称名詞は評価の対象外 (e.g. "ガ≒", "ヲ≒", "ニ≒", ...)
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
                for recall_top_k, _metric_name in self.analysis.items():
                    local_comp_result[key] = f"{pas_case}:{_metric_name}"
                    for rel in _gold_case_relations:
                        _topk = recall_top_k
                        if pas_case not in candidates:
                            break
                        elif len(candidates[pas_case]) < recall_top_k:
                            _topk = len(candidates[pas_case])

                        if rel.boundingBoxes is None:
                            # NOTE: Skip an out of span instance
                            continue
                        metrics.loc[pas_case, _metric_name].tp_fn += len(
                            rel.boundingBoxes
                        )
                        metrics.loc[pas_case, _metric_name].tp += self._eval_group_iou(
                            rel.boundingBoxes, candidates[pas_case][:_topk]
                        )

        self.comp_result.update({(sid, *k): v for k, v in local_comp_result.items()})
        return metrics

    @staticmethod
    def _eval_group_iou(
        gold_bboxes: list[BoundingBox], candidates: list[BoundingBoxPrediction]
    ) -> int:
        cnt = 0
        for gold_bbox in gold_bboxes:
            group_iou = [
                idx
                for idx, c in enumerate(candidates)
                if box_iou(gold_bbox.rect, c.rect) > IOU_THRESHOLD
            ]
            if len(group_iou) > 0:
                cnt += 1
        return cnt
