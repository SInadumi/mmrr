import hashlib
import logging
import os
import pickle
from pathlib import Path
from typing import Union

import numpy as np
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import ListConfig
from rhoknp import Document
from rhoknp.cohesion import ExophoraReferent, ExophoraReferentType
from tokenizers import Encoding
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy

from mmrr.datamodule.example import KyotoExample, SpecialTokenIndexer
from mmrr.tools.extractors import (
    BaseExtractor,
    BridgingExtractor,
    CoreferenceExtractor,
    PasExtractor,
)
from mmrr.tools.task import Task
from mmrr.utils.annotation import DatasetInfo
from mmrr.utils.dataset import CohesionBasePhrase, CohesionInputFeatures
from mmrr.utils.sub_document import (
    SequenceSplitter,
    SpanCandidate,
    to_orig_doc_id,
    to_sub_doc_id,
)
from mmrr.utils.util import IGNORE_INDEX, softmax

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class CohesionDataset(BaseDataset):
    def __init__(
        self,
        dataset_path: Union[str, Path],
        data_path: Union[str, Path],
        tasks: ListConfig,
        cases: ListConfig,
        bar_rels: ListConfig,
        max_seq_length: int,
        document_split_stride: int,
        tokenizer: PreTrainedTokenizerBase,
        exophora_referents: ListConfig,
        special_tokens: ListConfig,
        training: bool,
        flip_reader_writer: bool,
    ) -> None:
        super().__init__(
            dataset_path=Path(dataset_path),
            data_path=Path(data_path),
            cases=list(cases),
            max_seq_length=max_seq_length,
            tasks=[Task(task) for task in tasks],
            tokenizer=tokenizer,
            training=training,
        )

        self.exophora_referents: list[ExophoraReferent] = [
            ExophoraReferent(s) for s in exophora_referents
        ]
        self.special_tokens: list[str] = list(special_tokens)
        self.bar_rels: list[str] = list(bar_rels)
        self.flip_reader_writer: bool = flip_reader_writer
        self.is_jcre3_dataset = self.data_path.parts[-2] == "jcre3"

        exophora_referent_types: list[ExophoraReferentType] = [
            er.type for er in self.exophora_referents
        ]
        self.task_to_extractor: dict[Task, BaseExtractor] = {
            Task.PAS_ANALYSIS: PasExtractor(
                self.cases,
                exophora_referent_types,
                verbal_predicate=True,
                nominal_predicate=True,
            ),
            Task.BRIDGING_REFERENCE_RESOLUTION: BridgingExtractor(
                self.bar_rels, exophora_referent_types
            ),
            Task.COREFERENCE_RESOLUTION: CoreferenceExtractor(exophora_referent_types),
        }
        self.task_to_rels: dict[Task, list[str]] = {
            Task.PAS_ANALYSIS: self.cases,
            Task.BRIDGING_REFERENCE_RESOLUTION: self.bar_rels,
            Task.COREFERENCE_RESOLUTION: ["="],
        }

        # load knp format documents
        self.orig_documents: list[Document] = self.load_documents(self.data_path)
        self.doc_id2document: dict[str, Document] = {}
        for orig_document in self.orig_documents:
            self.doc_id2document.update(
                {
                    document.doc_id: document
                    for document in self._split_document(
                        document=orig_document,
                        max_token_length=max_seq_length
                        - len(tokenizer.additional_special_tokens)
                        - 2,  # -2: [CLS] and [SEP]
                        stride=document_split_stride,
                    )
                }
            )

        self.special_encoding: Encoding = self.tokenizer(
            self.special_tokens,
            is_split_into_words=True,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=False,
            add_special_tokens=False,
        ).encodings[0]

        self.examples: list[KyotoExample] = self._load_examples(
            self.documents, str(data_path)
        )

    # getter for knp documents
    @property
    def documents(self) -> list[Document]:
        return list(self.doc_id2document.values())

    @property
    def rel_types(self) -> list[str]:
        return [rel_type for task in self.tasks for rel_type in self.task_to_rels[task]]

    def _split_document(
        self, document: Document, max_token_length: int, stride: int
    ) -> list[Document]:
        sentence_tokens = [
            self._get_tokenized_len(sentence) for sentence in document.sentences
        ]
        if sum(sentence_tokens) <= max_token_length:
            return [document]

        splitter = SequenceSplitter(sentence_tokens, max_token_length, stride)
        sub_documents: list[Document] = []
        sub_idx = 0
        for span in splitter.split_into_spans():
            assert isinstance(span, SpanCandidate)
            sentences = document.sentences[span.start : span.end]
            sub_document = Document.from_sentences(sentences)
            sub_doc_id = to_sub_doc_id(document.doc_id, sub_idx, stride=span.stride)
            sub_document.doc_id = sub_doc_id
            for sentence, sub_sentence in zip(sentences, sub_document.sentences):
                sub_sentence.comment = sentence.comment
            sub_documents.append(sub_document)
            sub_idx += 1
        return sub_documents

    def _load_examples(
        self, documents: list[Document], documents_path: str
    ) -> list[KyotoExample]:
        examples = []
        load_cache: bool = (
            "DISABLE_CACHE" not in os.environ and "OVERWRITE_CACHE" not in os.environ
        )
        save_cache: bool = "DISABLE_CACHE" not in os.environ
        cohesion_cache_dir: Path = Path(
            os.environ.get("CACHE_DIR", f'/tmp/{os.environ["USER"]}/cohesion_cache'),
        )
        for document in tqdm(documents, desc="Loading examples", dynamic_ncols=True):
            # give enough options to identify examples
            hash_ = self._hash(
                documents_path,
                self.tasks,
                self.task_to_rels,
                self.task_to_extractor,
                self.flip_reader_writer,
            )
            example_cache_path = cohesion_cache_dir / hash_ / f"{document.doc_id}.pkl"
            if example_cache_path.exists() and load_cache:
                with example_cache_path.open(mode="rb") as f:
                    try:
                        example = pickle.load(f)
                    except EOFError:
                        example = self._load_example_from_document(document)
            else:
                example = self._load_example_from_document(document)
                if save_cache:
                    self._save_cache(example, example_cache_path)  # type: ignore
            examples.append(example)
        examples = self._post_process_examples(examples)
        if len(examples) == 0:
            logger.error(
                "No examples to process. "
                f"Make sure there exist any documents in {self.data_path} and they are not too long.",
            )
        return examples

    @rank_zero_only
    def _save_cache(self, example, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open(mode="wb") as f:
            pickle.dump(example, f)

    @staticmethod
    def _hash(*args) -> str:
        string = "".join(repr(a) for a in args)
        return hashlib.md5(string.encode()).hexdigest()

    def _post_process_examples(
        self, examples: list[KyotoExample]
    ) -> list[KyotoExample]:
        filtered = []
        for idx, example in enumerate(examples):
            phrases = next(iter(example.phrases.values()))
            morphemes = [
                morpheme for phrase in phrases for morpheme in phrase.morphemes
            ]
            encoding: Encoding = self.tokenizer(
                " ".join(morphemes),
                is_split_into_words=False,
                padding=PaddingStrategy.DO_NOT_PAD,
                truncation=False,
            ).encodings[0]
            if len(encoding.ids) > self.max_seq_length - len(self.special_tokens):
                continue
            special_token_indexer = SpecialTokenIndexer(
                self.special_tokens, len(encoding.ids), len(morphemes)
            )
            example.encoding = encoding
            example.special_token_indexer = special_token_indexer
            example.example_id = idx
            filtered.append(example)
        return filtered

    def _load_example_from_document(self, document: Document) -> KyotoExample:
        orig_doc_id = to_orig_doc_id(document.doc_id)
        sid_to_type_id: dict[str, int] = {}
        if self.is_jcre3_dataset:
            jcre3_dataset_dir = self.dataset_path / "jcre3" / "recording"
            dataset_info = DatasetInfo.from_json(
                (jcre3_dataset_dir / orig_doc_id / "info.json").read_text()
            )
            for utterance in dataset_info.utterances:
                assert utterance.speaker in ("主人", "ロボット")
                # 主人（著者） -> type_id=0, ロボット（読者） -> type_id=1
                sid_to_type_id.update(
                    {
                        sid: int(utterance.speaker == "ロボット")
                        for sid in utterance.sids
                    }
                )
        example = KyotoExample()
        example.load(
            document,
            tasks=self.tasks,
            task_to_extractor=self.task_to_extractor,
            task_to_rels=self.task_to_rels,
            sid_to_type_id=sid_to_type_id,
            flip_writer_reader_according_to_type_id=self.flip_reader_writer,
        )
        return example

    def dump_relation_prediction(
        self,
        relation_logits: np.ndarray,  # (rel, seq, seq), subword level
        example: KyotoExample,
    ) -> np.ndarray:  # (phrase, rel, phrase+special)
        """1 example 中に存在する基本句それぞれに対してシステム予測のリストを返す．"""
        predictions: list[np.ndarray] = []
        task_and_rels = [
            (task, rel) for task in self.tasks for rel in self.task_to_rels[task]
        ]
        assert len(relation_logits) == len(task_and_rels) == len(self.rel_types)
        assert example.special_token_indexer is not None
        for (task, _), logits in zip(task_and_rels, relation_logits):
            predictions.append(
                self._token_to_phrase_level(
                    logits,
                    example.phrases[task],
                    example.encoding,
                    example.special_token_indexer,
                )
            )
        return np.array(predictions).transpose(1, 0, 2)  # (phrase, rel, phrase+special)

    def _token_to_phrase_level(
        self,
        token_level_logits_matrix: np.ndarray,  # (seq, seq)
        phrases: list[CohesionBasePhrase],
        encoding: Encoding,
        special_token_indexer: SpecialTokenIndexer,
    ) -> np.ndarray:  # (phrase, phrase+special)
        phrase_level_scores_matrix: list[np.ndarray] = []
        for phrase in phrases:
            token_index_span: tuple[int, int] = encoding.word_to_tokens(
                phrase.head_morpheme_global_index
            )
            # Use the head subword as the representative of the source word.
            # Cast to built-in list because list operation is faster than numpy array operation.
            token_level_logits: list[float] = token_level_logits_matrix[
                token_index_span[0]
            ].tolist()  # (seq)
            phrase_level_logits: list[float] = []
            for target_phrase in phrases:
                # tgt 側は複数のサブワードから構成されるため平均を取る
                token_index_span = encoding.word_to_tokens(
                    target_phrase.head_morpheme_global_index
                )
                sliced_token_level_logits: list[float] = token_level_logits[
                    slice(*token_index_span)
                ]
                phrase_level_logits.append(
                    sum(sliced_token_level_logits) / len(sliced_token_level_logits)
                )
            phrase_level_logits += [
                token_level_logits[idx]
                for idx in special_token_indexer.token_level_indices
            ]
            assert len(phrase_level_logits) == len(phrases) + len(
                special_token_indexer.token_level_indices
            )
            phrase_level_scores_matrix.append(softmax(np.array(phrase_level_logits)))
        return np.array(phrase_level_scores_matrix)

    def _convert_example_to_feature(
        self, example: KyotoExample
    ) -> CohesionInputFeatures:
        """Loads a data file into a list of input features (token level)"""
        cohesion_labels: list[list[list[float]]] = []  # (rel, src, tgt)
        cohesion_mask: list[list[list[bool]]] = []  # (rel, src, tgt)
        assert example.special_token_indexer is not None
        for task in self.tasks:
            for rel in self.task_to_rels[task]:
                cohesion_labels.append(
                    self._convert_annotation_to_rel_labels(
                        example.phrases[task],
                        rel,
                        example.encoding,
                        example.special_token_indexer,
                    )
                )
                cohesion_mask.append(
                    self._convert_annotation_to_rel_mask(
                        example.phrases[task],
                        example.encoding,
                        example.special_token_indexer,
                    )
                )  # False -> mask, True -> keep

        assert example.encoding is not None, "encoding isn't set"

        source_mask = [False] * self.max_seq_length
        for global_index in example.analysis_target_morpheme_indices:
            for token_index in range(*example.encoding.word_to_tokens(global_index)):
                source_mask[token_index] = True

        source_label: list[list[int]] = [
            [IGNORE_INDEX] * self.max_seq_length for _ in range(len(self.tasks))
        ]  # (task, src)
        for task_index, task in enumerate(self.tasks):
            for phrase in example.phrases[task]:
                token_index_span: tuple[int, int] = example.encoding.word_to_tokens(
                    phrase.head_morpheme_global_index
                )
                for token_index in range(*token_index_span):
                    source_label[task_index][token_index] = int(phrase.is_target)

        padding_encoding: Encoding = self.tokenizer(
            "",
            add_special_tokens=False,
            padding=PaddingStrategy.MAX_LENGTH,
            truncation=False,
            max_length=self.max_seq_length
            - len(example.encoding.ids)
            - len(self.special_tokens),
        ).encodings[0]
        merged_encoding: Encoding = Encoding.merge(
            [example.encoding, self.special_encoding, padding_encoding]
        )
        return CohesionInputFeatures(
            example_id=example.example_id,
            input_ids=merged_encoding.ids,
            attention_mask=merged_encoding.attention_mask,
            token_type_ids=merged_encoding.type_ids,
            subword_map=self._generate_subword_map(
                merged_encoding.word_ids,
                example.encoding,
                example.special_token_indexer,
            ),
            source_mask=source_mask,
            target_mask=cohesion_mask,
            source_label=source_label,
            target_label=cohesion_labels,
        )

    def _generate_subword_map(
        self,
        word_ids: list[Union[int, None]],
        encoding: Encoding,
        special_token_indexer: SpecialTokenIndexer,
        include_special_tokens: bool = True,
    ) -> list[list[bool]]:  # (seq, seq)
        subword_map = [
            [False] * self.max_seq_length for _ in range(self.max_seq_length)
        ]
        for token_index, word_id in enumerate(word_ids):
            if (
                word_id is None
                or token_index in special_token_indexer.token_level_indices
            ):
                continue
            for token_id in range(*encoding.word_to_tokens(word_id)):
                subword_map[token_index][token_id] = True
        if include_special_tokens is True:
            for token_index in special_token_indexer.token_level_indices:
                subword_map[token_index][token_index] = True
        return subword_map

    def _convert_annotation_to_rel_labels(
        self,
        cohesion_base_phrases: list[CohesionBasePhrase],
        rel_type: str,
        encoding: Encoding,
        special_token_indexer: SpecialTokenIndexer,
    ) -> list[list[float]]:
        rel_labels = [[0.0] * self.max_seq_length for _ in range(self.max_seq_length)]
        for cohesion_base_phrase in cohesion_base_phrases:
            # use the head subword as the representative of the source word
            source_index_span = encoding.word_to_tokens(
                cohesion_base_phrase.head_morpheme_global_index
            )
            if cohesion_base_phrase.is_target is False:
                continue
            # 学習・解析対象基本句
            assert cohesion_base_phrase.rel2tags is not None
            for tag in cohesion_base_phrase.rel2tags[rel_type]:
                if tag in self.special_tokens:
                    token_index = special_token_indexer.get_token_level_index(tag)
                    rel_labels[source_index_span[0]][token_index] = 1.0
                else:
                    target_index_span: tuple[int, int] = encoding.word_to_tokens(
                        cohesion_base_phrases[int(tag)].head_morpheme_global_index,
                    )
                    for token_index in range(*target_index_span):
                        rel_labels[source_index_span[0]][token_index] = 1.0
        return rel_labels

    def _convert_annotation_to_rel_mask(
        self,
        cohesion_base_phrases: list[CohesionBasePhrase],
        encoding: Encoding,
        special_token_indexer: SpecialTokenIndexer,
    ) -> list[list[bool]]:
        rel_mask = [[False] * self.max_seq_length for _ in range(self.max_seq_length)]
        for cohesion_base_phrase in cohesion_base_phrases:
            # use the head subword as the representative of the source word
            source_index_span = encoding.word_to_tokens(
                cohesion_base_phrase.head_morpheme_global_index
            )
            for candidate in cohesion_base_phrase.referent_candidates:
                target_index_span = encoding.word_to_tokens(
                    candidate.head_morpheme_global_index
                )
                for token_index in range(*target_index_span):
                    rel_mask[source_index_span[0]][token_index] = True
            for special_token_global_index in special_token_indexer.token_level_indices:
                rel_mask[source_index_span[0]][special_token_global_index] = True
        return rel_mask

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> CohesionInputFeatures:
        return self._convert_example_to_feature(self.examples[idx])
