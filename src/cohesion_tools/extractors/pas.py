from collections import defaultdict
from typing import Dict, List, Optional

from rhoknp import BasePhrase, Document, Sentence
from rhoknp.cohesion import (
    Argument,
    EndophoraArgument,
    ExophoraArgument,
    ExophoraReferentType,
)

from .base import BaseExtractor, T


class PasExtractor(BaseExtractor):
    def __init__(
        self,
        cases: List[str],
        exophora_referent_types: List[ExophoraReferentType],
        verbal_predicate: bool = True,
        nominal_predicate: bool = True,
    ) -> None:
        super().__init__(exophora_referent_types)
        self.cases: List[str] = cases
        self.verbal_predicate: bool = verbal_predicate
        self.nominal_predicate: bool = nominal_predicate

    def extract_rels(self, predicate: BasePhrase) -> Dict[str, List[Argument]]:
        all_arguments: Dict[str, List[Argument]] = defaultdict(list)
        candidates: List[BasePhrase] = self.get_candidates(
            predicate, predicate.document.base_phrases
        )
        for case in self.cases:
            for argument in predicate.pas.get_arguments(case, relax=False):
                if isinstance(argument, EndophoraArgument):
                    if argument.base_phrase in candidates:
                        all_arguments[case].append(argument)
                elif isinstance(argument, ExophoraArgument):
                    if argument.exophora_referent.type in self.exophora_referent_types:
                        all_arguments[case].append(argument)
                else:
                    raise ValueError(
                        f"argument type {type(argument)} is not supported."
                    )
        return all_arguments

    def is_target(self, base_phrase: BasePhrase) -> bool:
        return self.is_pas_target(
            base_phrase, verbal=self.verbal_predicate, nominal=self.nominal_predicate
        )

    @staticmethod
    def is_pas_target(base_phrase: BasePhrase, verbal: bool, nominal: bool) -> bool:
        if verbal and "用言" in base_phrase.features:
            return True
        if nominal and "非用言格解析" in base_phrase.features:
            return True
        return False

    @staticmethod
    def is_candidate(unit: T, predicate: T) -> bool:
        is_anaphora = unit.global_index < predicate.global_index
        is_intra_sentential_cataphora = (
            unit.global_index > predicate.global_index
            and unit.sentence.sid == predicate.sentence.sid
        )
        return is_anaphora or is_intra_sentential_cataphora

    @staticmethod
    def restore_pas_annotation(document: Document) -> Document:
        for target_phrase in document.base_phrases:
            if not PasExtractor.is_pas_target(target_phrase, verbal=True, nominal=True):
                continue
            # 判定詞はスキップ
            if target_phrase.features.get("用言") == "判":
                continue
            arguments: Dict[str, list] = target_phrase.pas.get_all_arguments()
            # すでにアノテーションが存在する場合はスキップ
            if any(len(args) > 0 for args in arguments.values()):
                continue

            coreferring_base_phrase: BasePhrase = target_phrase
            coreferent_has_pas_annotation = False
            empty_rel_coreferent_global_indices: List[int] = []
            while coreferent_has_pas_annotation is False:
                result = PasExtractor._get_coreferent_by_rel_tags(
                    coreferring_base_phrase
                )
                if result is None:
                    break
                coreferring_base_phrase = result
                args_coref = coreferring_base_phrase.pas.get_all_arguments()
                if all(len(args) == 0 for args in args_coref.values()):
                    if (
                        coreferring_base_phrase.global_index
                        in empty_rel_coreferent_global_indices
                    ):
                        break  # circular coreference
                    empty_rel_coreferent_global_indices.append(
                        coreferring_base_phrase.global_index
                    )
                    continue
                coreferent_has_pas_annotation = True

            if coreferent_has_pas_annotation is True:
                assert coreferring_base_phrase is not None
                for rel_to_copy in coreferring_base_phrase.rel_tags:
                    if rel_to_copy.type in (
                        "ガ",
                        "ヲ",
                        "ニ",
                        "ガ２",
                        "デ",
                        "ト",
                        "カラ",
                        "ヨリ",
                        "マデ",
                        "ヘ",
                        "時間",
                        "外の関係",
                    ):
                        target_phrase.rel_tags.append(rel_to_copy)

        return document.reparse()

    @staticmethod
    def _get_coreferent_by_rel_tags(base_phrase: BasePhrase) -> Optional[BasePhrase]:
        """入力した基本句に共参照タグが付与されていた場合共参照先の基本句を返却．

        共参照タグが付与されていない場合，共参照先の基本句がない場合はNoneを返却．

        Args:
            base_phrase (BasePhrase): 共参照しているか調べたい基本句

        Returns:
            Optional[BasePhrase]: 共参照している基本句
        """
        sid_to_sentence: Dict[str, Sentence] = {
            sentence.sid: sentence for sentence in base_phrase.document.sentences
        }
        for rel in base_phrase.rel_tags:
            if rel.type == "=":
                if rel.sid is None:
                    continue
                assert rel.base_phrase_index is not None
                coreferent = sid_to_sentence[rel.sid].base_phrases[
                    rel.base_phrase_index
                ]
                if coreferent != base_phrase:
                    return coreferent
        return None
