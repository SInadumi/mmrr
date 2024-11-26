from abc import ABC, abstractmethod
from typing import Any, Optional

from rhoknp import Document
from tokenizers import Encoding

from mmrr.tools.extractors.base import BaseExtractor
from mmrr.tools.task import Task
from mmrr.utils.sub_document import extract_target_sentences


class BaseExample(ABC):
    def __init__(self) -> None:
        self.example_id: int = -1
        self.doc_id: str = ""
        self.analysis_target_morpheme_indices: list[int] = []
        self.encoding: Optional[Encoding] = None

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
