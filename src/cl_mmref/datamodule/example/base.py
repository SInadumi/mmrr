from typing import Optional

from rhoknp import Document
from tokenizers import Encoding

from cl_mmref.utils.sub_document import extract_target_sentences


class BaseExample:
    def __init__(self) -> None:
        self.example_id: int = -1
        self.analysis_target_morpheme_indices: list[int] = []
        self.encoding: Optional[Encoding] = None

    def load(self):
        raise NotImplementedError

    def set_knp_params(self, document: Document):
        analysis_target_morpheme_indices = []
        for sentence in extract_target_sentences(document.sentences):
            analysis_target_morpheme_indices += [
                m.global_index for m in sentence.morphemes
            ]
        self.analysis_target_morpheme_indices = analysis_target_morpheme_indices
