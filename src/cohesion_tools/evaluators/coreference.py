import copy
from typing import Callable, Collection, Dict, List, Optional, Set

import pandas as pd
from rhoknp import BasePhrase, Document
from rhoknp.cohesion import ExophoraReferent, ExophoraReferentType

from cohesion_tools.evaluators.utils import F1Metric


class CoreferenceResolutionEvaluator:
    """共参照解析の評価を行うクラス

    To evaluate system output with this class, you have to prepare gold data and system prediction data as instances of
    :class:`rhoknp.Document`

    Args:
        exophora_referent_types: 評価の対象とする外界照応の照応先 (rhoknp.cohesion.ExophoraReferentTypeType を参照)
        is_target_mention: 評価の対象とする基本句を指定する関数．引数に基本句を受け取り，対象とする基本句であれば True を返却．
    """

    def __init__(
        self,
        exophora_referent_types: Collection[ExophoraReferentType],
        is_target_mention: Optional[Callable[[BasePhrase], bool]] = None,
    ) -> None:
        self.exophora_referent_types: List[ExophoraReferentType] = list(exophora_referent_types)
        self.is_target_mention: Callable[[BasePhrase], bool] = is_target_mention or (lambda _: True)
        self.comp_result: Dict[tuple, str] = {}

    def run(self, predicted_document: Document, gold_document: Document) -> pd.Series:
        """Compute coreference resolution scores"""
        assert len(predicted_document.base_phrases) == len(gold_document.base_phrases)
        metrics: Dict[str, F1Metric] = {anal: F1Metric() for anal in ("endophora", "exophora")}
        local_comp_result: Dict[tuple, str] = {}
        for predicted_mention, gold_mention in zip(predicted_document.base_phrases, gold_document.base_phrases):
            if self.is_target_mention(predicted_mention) is True:
                predicted_other_mentions = self._filter_mentions(predicted_mention.get_coreferents(), predicted_mention)
                predicted_exophora_referents = self._filter_exophora_referents(
                    [e.exophora_referent for e in predicted_mention.entities if e.exophora_referent is not None],
                )
            else:
                predicted_other_mentions = []
                predicted_exophora_referents = set()

            if self.is_target_mention(gold_mention) is True:
                gold_other_mentions = self._filter_mentions(
                    gold_mention.get_coreferents(include_nonidentical=False),
                    gold_mention,
                )
                relaxed_gold_other_mentions = self._filter_mentions(
                    gold_mention.get_coreferents(include_nonidentical=True),
                    gold_mention,
                )
                gold_exophora_referents = self._filter_exophora_referents(
                    [e.exophora_referent for e in gold_mention.entities if e.exophora_referent is not None],
                )
                relaxed_gold_exophora_referents = self._filter_exophora_referents(
                    [e.exophora_referent for e in gold_mention.entities_all if e.exophora_referent is not None],
                )
            else:
                gold_other_mentions = relaxed_gold_other_mentions = []
                gold_exophora_referents = relaxed_gold_exophora_referents = set()

            key = (predicted_mention.global_index, "=")

            # Compute precision
            if predicted_other_mentions or predicted_exophora_referents:
                if any(mention in relaxed_gold_other_mentions for mention in predicted_other_mentions):
                    analysis = "endophora"
                    local_comp_result[key] = analysis
                    metrics[analysis].tp += 1
                elif predicted_exophora_referents & relaxed_gold_exophora_referents:
                    analysis = "exophora"
                    local_comp_result[key] = analysis
                    metrics[analysis].tp += 1
                else:
                    analysis = "endophora" if predicted_other_mentions else "exophora"
                    local_comp_result[key] = "wrong"
                metrics[analysis].tp_fp += 1

            # Compute recall
            if (
                gold_other_mentions
                or gold_exophora_referents
                or local_comp_result.get(key) in ("endophora", "exophora")
            ):
                if any(mention in relaxed_gold_other_mentions for mention in predicted_other_mentions):
                    analysis = "endophora"
                    assert local_comp_result[key] == analysis
                elif predicted_exophora_referents & relaxed_gold_exophora_referents:
                    analysis = "exophora"
                    assert local_comp_result[key] == analysis
                else:
                    analysis = "endophora" if gold_other_mentions else "exophora"
                    local_comp_result[key] = "wrong"
                metrics[analysis].tp_fn += 1
        self.comp_result.update({(gold_document.doc_id, *k): v for k, v in local_comp_result.items()})
        return pd.Series(metrics)

    @staticmethod
    def _filter_mentions(other_mentions: List[BasePhrase], mention: BasePhrase) -> List[BasePhrase]:
        """Filter out cataphora mentions"""
        return [
            another_mention for another_mention in other_mentions if another_mention.global_index < mention.global_index
        ]

    def _filter_exophora_referents(self, exophora_referents: List[ExophoraReferent]) -> Set[str]:
        filtered = set()
        for orig_exophora_referent in exophora_referents:
            exophora_referent = copy.copy(orig_exophora_referent)
            exophora_referent.index = None
            if exophora_referent.type in self.exophora_referent_types:
                filtered.add(exophora_referent.text)
        return filtered
