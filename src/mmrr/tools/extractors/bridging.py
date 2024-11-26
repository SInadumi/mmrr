from collections import defaultdict

from rhoknp import BasePhrase
from rhoknp.cohesion import (
    Argument,
    EndophoraArgument,
    ExophoraArgument,
    ExophoraReferentType,
)

from .base import BaseExtractor, T, U


class BridgingExtractor(BaseExtractor):
    def __init__(
        self, rel_types: list[str], exophora_referent_types: list[ExophoraReferentType]
    ) -> None:
        super().__init__(exophora_referent_types)
        assert "ノ" in rel_types, '"ノ" not found in rel_types'
        self.rel_types = rel_types

    def extract_rels(self, base_phrase: T) -> dict[str, list[Argument]]:
        assert isinstance(base_phrase, BasePhrase)  # `base_phrase` means anaphor
        all_referents: dict[str, list[Argument]] = defaultdict(list)
        candidates: list[BasePhrase] = self.get_candidates(
            base_phrase, base_phrase.document.base_phrases
        )
        for rel_type in self.rel_types:
            for referent in base_phrase.pas.get_arguments(rel_type, relax=False):
                if isinstance(referent, EndophoraArgument):
                    if referent.base_phrase in candidates:
                        all_referents[rel_type].append(referent)
                elif isinstance(referent, ExophoraArgument):
                    if referent.exophora_referent.type in self.exophora_referent_types:
                        all_referents[rel_type].append(referent)
                else:
                    raise ValueError(
                        f"argument type {type(referent)} is not supported."
                    )
        return all_referents

    def is_target(self, anaphor: T) -> bool:
        assert isinstance(anaphor, BasePhrase)
        return self.is_bridging_target(anaphor)

    @staticmethod
    def is_bridging_target(anaphor: BasePhrase) -> bool:
        return (
            anaphor.features.get("体言") is True
            and "非用言格解析" not in anaphor.features
        )

    @staticmethod
    def is_candidate(unit: U, anaphor: U) -> bool:
        is_anaphora = unit.global_index < anaphor.global_index
        is_intra_sentential_cataphora = (
            unit.global_index > anaphor.global_index
            and unit.sentence.sid == anaphor.sentence.sid
        )
        return is_anaphora or is_intra_sentential_cataphora
