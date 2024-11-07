import pandas as pd

from cl_mmref.utils.annotation import BoundingBox, PhraseAnnotation
from cl_mmref.utils.prediction import BoundingBoxPrediction, PhrasePrediction
from cl_mmref.utils.util import box_iou

from ..constants import IOU_THRESHOLD, RECALL_TOP_KS
from .utils import F1Metric


class VisCoreferenceResolutionEvaluator:
    """A class to evaluate visual coreference resolution or phrase grounding"""

    def __init__(self) -> None:
        self.comp_result: dict[tuple, str] = {}
        # NOTE: 現在の実装では総称名詞は評価の対象外 ("=≒")
        self.rel: str = "="
        self.analysis: dict[int, str] = {k: f"recall@{k}" for k in RECALL_TOP_KS}

    def run(
        self,
        sid: str,
        predicted_phrases: PhrasePrediction,
        gold_phrases: PhraseAnnotation,
    ) -> pd.DataFrame:
        assert len(predicted_phrases) == len(gold_phrases)
        metrics = pd.DataFrame(
            [[F1Metric() for _ in self.analysis.values()] for _ in [self.rel]],
            index=[self.rel],
            columns=list(self.analysis.values()),
        )

        local_comp_result: dict[tuple, str] = {}
        for idx, (predicted_mention, gold_mention) in enumerate(
            zip(predicted_phrases, gold_phrases)
        ):
            _gold_coref_relations = [
                rel for rel in gold_mention.relations if rel.type == self.rel
            ]
            if len(_gold_coref_relations) == 0:
                continue

            # NOTE: `is_target==False` ならば `len(candidates)==0`
            candidates: list[BoundingBoxPrediction] = []
            if len(predicted_mention.relations) == 1:
                assert predicted_mention.relations[0].type == self.rel
                candidates = predicted_mention.relations[0].bounding_boxes
            elif len(predicted_mention.relations) > 1:
                raise ValueError

            key = (f"{idx}:{predicted_mention.text}", self.rel)

            # Compute recall
            for recall_top_k, _metric_name in self.analysis.items():
                local_comp_result[key] = f"{self.rel}:{_metric_name}"
                for rel in _gold_coref_relations:
                    _topk = recall_top_k
                    if len(candidates) == 0:
                        break
                    elif len(candidates) < recall_top_k:
                        _topk = len(candidates)

                    if rel.boundingBoxes is None:
                        # NOTE: Skip an out of span instance
                        continue
                    metrics.loc[self.rel, _metric_name].tp_fn += len(rel.boundingBoxes)
                    metrics.loc[self.rel, _metric_name].tp += self._eval_group_iou(
                        rel.boundingBoxes, candidates[:_topk]
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
        assert cnt <= len(gold_bboxes)
        return cnt
