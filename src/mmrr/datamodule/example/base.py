from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Optional

from rhoknp import Document
from tokenizers import Encoding

from mmrr.tools.extractors.base import BaseExtractor
from mmrr.tools.task import Task
from mmrr.utils.sub_document import extract_target_sentences


class SpecialTokenIndexer:
    def __init__(
        self, special_tokens: list[str], num_tokens: int, num_morphemes: int
    ) -> None:
        self.special_tokens: list[str] = special_tokens
        self._special_token2token_level_index: dict[str, int] = {
            st: num_tokens + i for i, st in enumerate(special_tokens)
        }
        self._special_token2morpheme_level_index: dict[str, int] = {
            st: num_morphemes + i for i, st in enumerate(special_tokens)
        }

    def get_morpheme_level_index(self, special_token: str) -> int:
        return self._special_token2morpheme_level_index[special_token]

    def get_token_level_index(self, special_token: str) -> int:
        return self._special_token2token_level_index[special_token]

    @cached_property
    def get_morpheme_level_indices(self) -> list[int]:
        return list(self._special_token2morpheme_level_index.values())

    @cached_property
    def token_level_indices(self) -> list[int]:
        return list(self._special_token2token_level_index.values())

    @cached_property
    def token_and_morpheme_level_indices(self) -> list[tuple[int, int]]:
        return list(
            zip(
                self._special_token2token_level_index.values(),
                self._special_token2morpheme_level_index.values(),
            )
        )


class BaseExample(ABC):
    def __init__(self) -> None:
        self.example_id: int = -1
        self.doc_id: str = ""
        self.analysis_target_morpheme_indices: list[int] = []
        self.encoding: Optional[Encoding] = None
        self.special_token_indexer: Optional[SpecialTokenIndexer] = None

    @abstractmethod
    def load(
        self,
        document: Document,
        tasks: list[Task],
        task_to_extractor: dict[Task, BaseExtractor],
        *args: Any,
        **kwargs: Any,
    ):
        raise NotImplementedError

    def set_knp_params(self, document: Document):
        self.doc_id = document.doc_id
        analysis_target_morpheme_indices = []
        for sentence in extract_target_sentences(document.sentences):
            analysis_target_morpheme_indices += [
                m.global_index for m in sentence.morphemes
            ]
        self.analysis_target_morpheme_indices = analysis_target_morpheme_indices
