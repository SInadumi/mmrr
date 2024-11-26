import copy
from typing import Callable, ClassVar, Collection, Optional

import pandas as pd
from rhoknp import Document
from rhoknp.cohesion import (
    Argument,
    ArgumentType,
    EndophoraArgument,
    ExophoraArgument,
    ExophoraReferentType,
    Predicate,
)

from .utils import F1Metric


class BridgingReferenceResolutionEvaluator:
    """橋渡し参照解析の評価を行うクラス

    To evaluate system output with this class, you have to prepare gold data and system prediction data as instances of
    :class:`rhoknp.Document`

    Args:
        exophora_referent_types: 評価の対象とする外界照応の照応先 (rhoknp.cohesion.ExophoraReferentTypeType を参照)
        rel_types: 橋渡し参照解析の評価の対象とする関係 (rhoknp.cohesion.rel.CASE_TYPES を参照)
        is_target_anaphor: 評価の対象とする照応詞を指定する関数．引数に照応詞を受け取り，対象とする照応詞であれば True を返却．
    """

    ARGUMENT_TYPE_TO_ANALYSIS_TYPE: ClassVar[dict[ArgumentType, str]] = {
        ArgumentType.CASE_EXPLICIT: "dep",
        ArgumentType.CASE_HIDDEN: "dep",
        ArgumentType.OMISSION: "zero_endophora",
        ArgumentType.EXOPHORA: "exophora",
    }

    def __init__(
        self,
        exophora_referent_types: Collection[ExophoraReferentType],
        rel_types: Collection[str],
        is_target_anaphor: Optional[Callable[[Predicate], bool]] = None,
    ) -> None:
        self.exophora_referent_types: list[ExophoraReferentType] = list(
            exophora_referent_types
        )
        self.rel_types: list[str] = list(rel_types)
        self.is_target_anaphor: Callable[[Predicate], bool] = is_target_anaphor or (
            lambda _: True
        )
        self.comp_result: dict[tuple, str] = {}

    def run(
        self, predicted_document: Document, gold_document: Document
    ) -> pd.DataFrame:
        """Compute bridging reference resolution scores"""
        metrics = pd.DataFrame(
            [
                [F1Metric() for _ in ("dep", "zero_endophora", "exophora")]
                for _ in self.rel_types
            ],
            index=self.rel_types,
            columns=["dep", "zero_endophora", "exophora"],
        )
        predicted_anaphors = [
            base_phrase.pas.predicate for base_phrase in predicted_document.base_phrases
        ]
        gold_anaphors = [
            base_phrase.pas.predicate for base_phrase in gold_document.base_phrases
        ]
        local_comp_result: dict[tuple, str] = {}

        assert len(predicted_anaphors) == len(gold_anaphors)
        for predicted_anaphor, gold_anaphor in zip(predicted_anaphors, gold_anaphors):
            for rel_type in self.rel_types:
                if self.is_target_anaphor(predicted_anaphor) is True:
                    predicted_antecedents: list[Argument] = self._filter_referents(
                        predicted_anaphor.pas.get_arguments(rel_type, relax=False),
                        predicted_anaphor,
                    )
                else:
                    predicted_antecedents = []
                # Assuming one antecedent for one anaphor
                assert len(predicted_antecedents) in (0, 1)

                if self.is_target_anaphor(gold_anaphor) is True:
                    gold_antecedents: list[Argument] = self._filter_referents(
                        gold_anaphor.pas.get_arguments(rel_type, relax=False),
                        gold_anaphor,
                    )
                    relaxed_gold_antecedents: list[Argument] = (
                        gold_anaphor.pas.get_arguments(
                            rel_type,
                            relax=True,
                            include_nonidentical=True,
                        )
                    )
                    if rel_type == "ノ":
                        relaxed_gold_antecedents += gold_anaphor.pas.get_arguments(
                            "ノ？", relax=True, include_nonidentical=True
                        )
                    relaxed_gold_antecedents = self._filter_referents(
                        relaxed_gold_antecedents, gold_anaphor
                    )
                else:
                    gold_antecedents = relaxed_gold_antecedents = []

                key = (predicted_anaphor.base_phrase.global_index, rel_type)

                # Compute precision
                if len(predicted_antecedents) > 0:
                    predicted_antecedent = predicted_antecedents[0]
                    if predicted_antecedent in relaxed_gold_antecedents:
                        relaxed_gold_antecedent = relaxed_gold_antecedents[
                            relaxed_gold_antecedents.index(predicted_antecedent)
                        ]
                        analysis = self.ARGUMENT_TYPE_TO_ANALYSIS_TYPE[
                            relaxed_gold_antecedent.type
                        ]
                        local_comp_result[key] = analysis
                        metrics.loc[rel_type, analysis].tp += 1
                    else:
                        analysis = self.ARGUMENT_TYPE_TO_ANALYSIS_TYPE[
                            predicted_antecedent.type
                        ]
                        local_comp_result[key] = "wrong"
                    metrics.loc[rel_type, analysis].tp_fp += 1

                # Compute recall
                if gold_antecedents or (
                    local_comp_result.get(key, None)
                    in self.ARGUMENT_TYPE_TO_ANALYSIS_TYPE.values()
                ):
                    recalled_antecedent: Optional[Argument] = None
                    for relaxed_gold_antecedent in relaxed_gold_antecedents:
                        if relaxed_gold_antecedent in predicted_antecedents:
                            recalled_antecedent = relaxed_gold_antecedent  # 予測されている先行詞を優先して正解の先行詞に採用
                            break
                    if recalled_antecedent is not None:
                        analysis = self.ARGUMENT_TYPE_TO_ANALYSIS_TYPE[
                            recalled_antecedent.type
                        ]
                        if analysis == "overt":
                            analysis = "dep"
                        assert local_comp_result[key] == analysis
                    else:
                        analysis = self.ARGUMENT_TYPE_TO_ANALYSIS_TYPE[
                            gold_antecedents[0].type
                        ]
                        if analysis == "overt":
                            analysis = "dep"
                        if len(predicted_antecedents) > 0:
                            assert local_comp_result[key] == "wrong"
                        else:
                            local_comp_result[key] = "wrong"
                    metrics.loc[rel_type, analysis].tp_fn += 1
        self.comp_result.update(
            {(gold_document.doc_id, *k): v for k, v in local_comp_result.items()}
        )
        return metrics

    def _filter_referents(
        self, referents: list[Argument], anaphor: Predicate
    ) -> list[Argument]:
        filtered = []
        for orig_referent in referents:
            referent = copy.copy(orig_referent)
            if referent.case.endswith("≒"):
                referent.case = referent.case[:-1]
            if referent.case == "ノ？":
                referent.case = "ノ"
            if isinstance(referent, ExophoraArgument):
                referent.exophora_referent.index = (
                    None  # 「不特定:人１」なども「不特定:人」として扱う
                )
                if referent.exophora_referent.type not in self.exophora_referent_types:
                    continue
            else:
                assert isinstance(referent, EndophoraArgument)
                # Filter out self-anaphora
                if referent.base_phrase == anaphor.base_phrase:
                    continue
                # Filter out cataphora
                if (
                    referent.base_phrase.global_index > anaphor.base_phrase.global_index
                    and referent.base_phrase.sentence.sid
                    != anaphor.base_phrase.sentence.sid
                ):
                    continue
            filtered.append(referent)
        return filtered
