import pandas as pd

from mmrr.utils.annotation import BoundingBox, PhraseAnnotation
from mmrr.utils.prediction import BoundingBoxPrediction, PhrasePrediction
from mmrr.utils.util import box_iou

from ..constants import IOU_THRESHOLD, RECALL_TOP_KS
from .utils import F1Metric


class MultiModalCoreferenceResolutionEvaluator:
    """A class to evaluate multimodal direct reference relation (coreference) analysis. This is also known as phrase grounding."""

    def __init__(self) -> None:
        self.comp_result: dict[tuple, str] = {}
        # NOTE: 現在の実装では総称名詞は評価の対象外 ("=≒")
        self.rel_type: str = "="
        self.analysis: dict[int, str] = {k: f"recall@{k}" for k in RECALL_TOP_KS}

    def run(
        self,
        sid: str,
        predicted_phrases: list[PhrasePrediction],
        gold_phrases: list[PhraseAnnotation],
    ) -> pd.DataFrame:
        assert len(predicted_phrases) == len(gold_phrases)
        metrics = pd.DataFrame(
            [[F1Metric() for _ in self.analysis.values()] for _ in [self.rel_type]],
            index=[self.rel_type],
            columns=list(self.analysis.values()),
        )

        local_comp_result: dict[tuple, str] = {}
        for idx, (predicted_mention, gold_mention) in enumerate(
            zip(predicted_phrases, gold_phrases)
        ):
            _gold_coref_relations = [
                rel for rel in gold_mention.relations if rel.type == self.rel_type
            ]
            if len(_gold_coref_relations) == 0:
                continue

            # NOTE: `is_target==False` ならば `len(candidates)==0`
            candidates: list[BoundingBoxPrediction] = []
            for relation in predicted_mention.relations:
                if relation.type == self.rel_type:
                    candidates = relation.bounding_boxes
            key = (f"{idx}:{predicted_mention.text}", self.rel_type)

            # Compute recall
            for recall_top_k, _metric_name in self.analysis.items():
                local_comp_result[key] = f"{self.rel_type}:{_metric_name}"
                for rel in _gold_coref_relations:
                    _topk = recall_top_k
                    if len(candidates) == 0:
                        break
                    elif len(candidates) < recall_top_k:
                        _topk = len(candidates)

                    if rel.boundingBoxes is None:
                        # NOTE: Skip an out of span instance
                        continue
                    metrics.loc[self.rel_type, _metric_name].tp_fn += len(
                        rel.boundingBoxes
                    )
                    metrics.loc[self.rel_type, _metric_name].tp += self._eval_group_iou(
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
