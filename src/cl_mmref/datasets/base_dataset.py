from pathlib import Path
from typing import Union

import numpy as np
import torch
from rhoknp import Document, Sentence
from rhoknp.utils.reader import chunk_by_document
from transformers import PreTrainedTokenizerBase

from cl_mmref.datamodule.example import KyotoExample, MMRefExample
from cl_mmref.utils.util import sigmoid

ExampleType = Union[KyotoExample, MMRefExample]


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        training: bool,
    ):
        super().__init__()
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.training: bool = training

    @staticmethod
    def load_documents(document_path: Path, ext: str = "knp") -> list[Document]:
        """Search KNP format files in the given path and load them into document object.

        If the path is a directory, search files in the directory. Each file is assumed to have one document.
        If the path is a file, load contents in the file. The file is assumed to have multiple documents.

        Args:
            document_path: Path to the directory or file.
            ext: Extension of the KNP format file. Default is "knp".

        Returns:
            List of document objects.
        """
        documents = []
        if document_path.is_dir():
            for path in sorted(document_path.glob(f"*.{ext}")):
                documents.append(Document.from_knp(path.read_text()))
        else:
            with document_path.open() as f:
                for knp_text in chunk_by_document(f):
                    documents.append(Document.from_knp(knp_text))
        return documents

    def _get_tokenized_len(self, source: Union[Document, Sentence]) -> int:
        return len(self.tokenizer.tokenize(" ".join(m.text for m in source.morphemes)))

    def dump_relation_prediction(self):
        raise NotImplementedError

    def dump_source_mask_prediction(
        self,
        token_level_source_mask_logits: np.ndarray,  # (task, seq)
        example: ExampleType,
    ) -> np.ndarray:  # (phrase, task)
        """1 example 中に存在する基本句それぞれに対してシステム予測のリストを返す．"""
        assert example.encoding is not None, "encoding isn't set"
        phrase_task_scores: list[list[float]] = []
        token_level_source_mask_scores = sigmoid(token_level_source_mask_logits)
        assert len(token_level_source_mask_scores) == len(self.tasks)
        for task, token_level_scores in zip(
            self.tasks, token_level_source_mask_scores.tolist()
        ):
            phrase_level_scores: list[float] = []
            for phrase in example.phrases[task]:
                token_index_span: tuple[int, int] = example.encoding.word_to_tokens(
                    phrase.head_morpheme_global_index
                )
                sliced_token_level_scores: list[float] = token_level_scores[
                    slice(*token_index_span)
                ]
                phrase_level_scores.append(
                    sum(sliced_token_level_scores) / len(sliced_token_level_scores)
                )
            phrase_task_scores.append(phrase_level_scores)
        return np.array(phrase_task_scores).transpose()
