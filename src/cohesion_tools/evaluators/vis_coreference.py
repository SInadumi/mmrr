import pandas as pd

from utils.annotation import PhraseAnnotation
from utils.prediction import BoundingBoxPrediction, PhrasePrediction

from .utils import RECALL_TOP_KS, F1Metric


class VisCoreferenceResolutionEvaluator:
    """A class to evaluate visual coreference resolution or phrase grounding"""

    def __init__(self) -> None:
        self.comp_result: dict[tuple, str] = {}
        # NOTE: 現在の実装では非総称名詞は評価の対象外 ("=≒")
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
            assert len(predicted_mention.relations) == 1
            assert predicted_mention.relations[0].type == self.rel

            candidates: list[BoundingBoxPrediction] = predicted_mention.relations[
                0
            ].bounding_box
            key = (f"{idx}:{predicted_mention.text}", self.rel)

            # Compute recall
            # 正解が複数ある場合、そのうち一つが当てられていればそれを正解に採用
            for recall_top_k, _metric_name in self.analysis.items():
                local_comp_result[key] = f"{self.rel}:{_metric_name}"
                metrics.loc[self.rel, _metric_name].tp_fn += 1
                for rel in _gold_coref_relations:
                    if rel.classId in set(
                        int(c.class_id) for c in candidates[:recall_top_k]
                    ):
                        metrics.loc[self.rel, _metric_name].tp += 1
                        break
        self.comp_result.update({(sid, *k): v for k, v in local_comp_result.items()})
        return metrics
