import copy
from typing import Callable, ClassVar, Collection, Dict, List, Optional

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


class PASAnalysisEvaluator:
    """述語項構造解析の評価を行うクラス

    To evaluate system output with this class, you have to prepare gold data and system prediction data as instances of
    :class:`rhoknp.Document`

    Args:
        exophora_referent_types: 評価の対象とする外界照応の照応先 (rhoknp.cohesion.ExophoraReferentTypeType を参照)
        cases: 述語項構造の評価の対象とする格 (rhoknp.cohesion.rel.CASE_TYPES を参照)
        is_target_predicate: 評価の対象とする述語を指定する関数．引数に述語を受け取り，対象とする述語であれば True を返却．
    """

    ARGUMENT_TYPE_TO_ANALYSIS_TYPE: ClassVar[Dict[ArgumentType, str]] = {
        ArgumentType.CASE_EXPLICIT: "overt",
        ArgumentType.CASE_HIDDEN: "dep",
        ArgumentType.OMISSION: "zero_endophora",
        ArgumentType.EXOPHORA: "exophora",
    }

    def __init__(
        self,
        exophora_referent_types: Collection[ExophoraReferentType],
        cases: Collection[str],
        is_target_predicate: Optional[Callable[[Predicate], bool]] = None,
    ) -> None:
        self.exophora_referent_types: List[ExophoraReferentType] = list(
            exophora_referent_types
        )
        self.cases: List[str] = list(cases)
        self.is_target_predicate: Callable[[Predicate], bool] = is_target_predicate or (
            lambda _: True
        )
        self.comp_result: Dict[tuple, str] = {}

    def run(
        self, predicted_document: Document, gold_document: Document
    ) -> pd.DataFrame:
        """Compute predicate-argument structure analysis scores"""
        metrics = pd.DataFrame(
            [
                [F1Metric() for _ in self.ARGUMENT_TYPE_TO_ANALYSIS_TYPE.values()]
                for _ in self.cases
            ],
            index=self.cases,
            columns=list(self.ARGUMENT_TYPE_TO_ANALYSIS_TYPE.values()),
        )
        predicted_predicates = [
            base_phrase.pas.predicate for base_phrase in predicted_document.base_phrases
        ]
        gold_predicates = [
            base_phrase.pas.predicate for base_phrase in gold_document.base_phrases
        ]
        local_comp_result: Dict[tuple, str] = {}

        assert len(predicted_predicates) == len(gold_predicates)
        for predicted_predicate, gold_predicate in zip(
            predicted_predicates, gold_predicates
        ):
            for pas_case in self.cases:
                if self.is_target_predicate(predicted_predicate) is True:
                    predicted_pas_arguments = self._filter_arguments(
                        predicted_predicate.pas.get_arguments(pas_case, relax=False),
                        predicted_predicate,
                    )
                else:
                    predicted_pas_arguments = []
                # Assuming one argument for one predicate
                assert len(predicted_pas_arguments) in (0, 1)

                if self.is_target_predicate(gold_predicate) is True:
                    gold_pas_arguments = self._filter_arguments(
                        gold_predicate.pas.get_arguments(pas_case, relax=False),
                        gold_predicate,
                    )
                    relaxed_gold_pas_arguments = gold_predicate.pas.get_arguments(
                        pas_case,
                        relax=True,
                        include_nonidentical=True,
                    )
                    if pas_case == "ガ":
                        relaxed_gold_pas_arguments += gold_predicate.pas.get_arguments(
                            "判ガ",
                            relax=True,
                            include_nonidentical=True,
                        )
                    relaxed_gold_pas_arguments = self._filter_arguments(
                        relaxed_gold_pas_arguments, gold_predicate
                    )
                else:
                    gold_pas_arguments = relaxed_gold_pas_arguments = []

                key = (predicted_predicate.base_phrase.global_index, pas_case)

                # Compute precision
                if len(predicted_pas_arguments) > 0:
                    predicted_pas_argument = predicted_pas_arguments[0]
                    if predicted_pas_argument in relaxed_gold_pas_arguments:
                        relaxed_gold_pas_argument = relaxed_gold_pas_arguments[
                            relaxed_gold_pas_arguments.index(predicted_pas_argument)
                        ]
                        # Use argument_type of gold argument if possible
                        analysis = self.ARGUMENT_TYPE_TO_ANALYSIS_TYPE[
                            relaxed_gold_pas_argument.type
                        ]
                        local_comp_result[key] = analysis
                        metrics.loc[pas_case, analysis].tp += 1
                    else:
                        analysis = self.ARGUMENT_TYPE_TO_ANALYSIS_TYPE[
                            predicted_pas_argument.type
                        ]
                        local_comp_result[key] = "wrong"  # precision が下がる
                    metrics.loc[pas_case, analysis].tp_fp += 1

                # Compute recall
                # 正解が複数ある場合、そのうち一つが当てられていればそれを正解に採用
                if (
                    len(gold_pas_arguments) > 0
                    or local_comp_result.get(key)
                    in self.ARGUMENT_TYPE_TO_ANALYSIS_TYPE.values()
                ):
                    recalled_pas_argument: Optional[Argument] = None
                    for relaxed_gold_pas_argument in relaxed_gold_pas_arguments:
                        if relaxed_gold_pas_argument in predicted_pas_arguments:
                            recalled_pas_argument = relaxed_gold_pas_argument  # 予測されている項を優先して正解の項に採用
                            break
                    if recalled_pas_argument is not None:
                        analysis = self.ARGUMENT_TYPE_TO_ANALYSIS_TYPE[
                            recalled_pas_argument.type
                        ]
                        assert local_comp_result[key] == analysis
                    else:
                        # いずれも当てられていなければ、relax されていない項から一つを選び正解に採用
                        analysis = self.ARGUMENT_TYPE_TO_ANALYSIS_TYPE[
                            gold_pas_arguments[0].type
                        ]
                        if len(predicted_pas_arguments) > 0:
                            assert local_comp_result[key] == "wrong"
                        else:
                            local_comp_result[key] = "wrong"  # recall が下がる
                    metrics.loc[pas_case, analysis].tp_fn += 1
        self.comp_result.update(
            {(gold_document.doc_id, *k): v for k, v in local_comp_result.items()}
        )
        return metrics

    def _filter_arguments(
        self, arguments: List[Argument], predicate: Predicate
    ) -> List[Argument]:
        filtered = []
        for orig_argument in arguments:
            argument = copy.copy(orig_argument)
            if argument.case.endswith("≒"):
                argument.case = argument.case[:-1]
            if argument.case == "判ガ":
                argument.case = "ガ"
            if isinstance(argument, ExophoraArgument):
                argument.exophora_referent.index = (
                    None  # 「不特定:人１」なども「不特定:人」として扱う
                )
                if argument.exophora_referent.type not in self.exophora_referent_types:
                    continue
            else:
                assert isinstance(argument, EndophoraArgument)
                # Filter out self-anaphora
                if argument.base_phrase == predicate.base_phrase:
                    continue
                # Filter out cataphora
                if (
                    argument.base_phrase.global_index
                    > predicate.base_phrase.global_index
                    and argument.base_phrase.sentence.sid
                    != predicate.base_phrase.sentence.sid
                ):
                    continue
            filtered.append(argument)
        return filtered
