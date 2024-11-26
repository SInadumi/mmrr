from typing import Union

from rhoknp import BasePhrase
from rhoknp.cohesion import ExophoraReferent, ExophoraReferentType

from .base import BaseExtractor, T, U


class CoreferenceExtractor(BaseExtractor):
    def __init__(self, exophora_referent_types: list[ExophoraReferentType]) -> None:
        super().__init__(exophora_referent_types)

    def extract_rels(self, base_phrase: T) -> list[Union[BasePhrase, ExophoraReferent]]:
        assert isinstance(base_phrase, BasePhrase)  # `base_phrase` means mention
        referents: list[Union[BasePhrase, ExophoraReferent]] = []
        candidates: list[BasePhrase] = self.get_candidates(
            base_phrase, base_phrase.document.base_phrases
        )
        for coreferent in base_phrase.get_coreferents(
            include_nonidentical=False, include_self=False
        ):
            if coreferent in candidates:
                referents.append(coreferent)
        for exophora_referent in [
            e.exophora_referent
            for e in base_phrase.entities
            if e.exophora_referent is not None
        ]:
            if exophora_referent.type in self.exophora_referent_types:
                referents.append(exophora_referent)
        return referents

    def is_target(self, mention: T) -> bool:
        assert isinstance(mention, BasePhrase)
        return self.is_coreference_target(mention)

    @staticmethod
    def is_coreference_target(mention: BasePhrase) -> bool:
        return mention.features.get("体言") is True

    @staticmethod
    def is_candidate(target_mention: U, source_mention: U) -> bool:
        return target_mention.global_index < source_mention.global_index
