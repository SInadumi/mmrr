from abc import ABC, abstractmethod
from typing import Any, Collection, List, TypeVar

from rhoknp import BasePhrase, Morpheme
from rhoknp.cohesion import ExophoraReferentType

T = TypeVar("T", BasePhrase, Morpheme)


class BaseExtractor(ABC):
    def __init__(self, exophora_referent_types: List[ExophoraReferentType]) -> None:
        self.exophora_referent_types = exophora_referent_types

    @abstractmethod
    def extract_rels(self, base_phrase: BasePhrase) -> Collection[Any]:
        raise NotImplementedError

    @abstractmethod
    def is_target(self, base_phrase: BasePhrase) -> bool:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def is_candidate(possible_candidate: T, anaphor: T) -> bool:
        raise NotImplementedError

    def get_candidates(self, anaphor: T, morphemes_or_base_phrases: Collection[T]) -> List[T]:
        return [unit for unit in morphemes_or_base_phrases if self.is_candidate(unit, anaphor) is True]

