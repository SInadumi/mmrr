import hashlib
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Union

import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import ListConfig
from rhoknp import Document
from rhoknp.cohesion import ExophoraReferent, ExophoraReferentType
from tokenizers import Encoding
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy

from cohesion_tools.extractors import MMRefExtractor
from cohesion_tools.extractors.base import BaseExtractor
from cohesion_tools.task import Task
from datamodule.example.mmref import MMRefExample
from utils.annotation import DatasetInfo, ImageTextAnnotation
from utils.dataset import (
    MMRefInputFeatures,
    MMRefBasePhrase,
    TextualFeatures,
    VisualFeatures,
    ObjectFeature
)
from utils.sub_document import to_orig_doc_id
from utils.util import IGNORE_INDEX

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class MMRefDataset(BaseDataset):
    def __init__(
        self,
        dataset_path: Union[str, Path],
        data_path: Union[str, Path],
        tasks: ListConfig,  # "vis-pas", "vis-coref"
        cases: ListConfig,  # target case frames
        max_seq_length: int,
        document_split_stride: int,
        tokenizer: PreTrainedTokenizerBase,
        exophora_referents: ListConfig,
        special_tokens: ListConfig,
        training: bool,
        flip_reader_writer: bool,
        image_input_width: int = 0,  # NOTE: 入力対象となる発話区間を "image_input_width" フレーム分拡張
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.data_path = Path(data_path)
        self.exophora_referents: list[ExophoraReferentType] = [
            ExophoraReferent(s) for s in exophora_referents
        ]
        self.special_tokens: list[str] = list(special_tokens)

        super().__init__(
            data_path=self.data_path,
            max_seq_length=max_seq_length,
            document_split_stride=document_split_stride,
            tokenizer=tokenizer,
            training=training,
        )
        self.tasks = [Task(task) for task in tasks]
        self.cases = cases
        self.special_to_index: dict[str, int] = {
            token: max_seq_length - len(self.special_tokens) + i
            for i, token in enumerate(self.special_tokens)
        }
        self.flip_reader_writer: bool = flip_reader_writer
        self.image_input_width = image_input_width
        self.is_jcre3_dataset = self.data_path.parts[-2] == "jcre3"

        exophora_referent_types: list[ExophoraReferentType] = [
            er.type for er in self.exophora_referents
        ]
        self.task_to_extractor: dict[Task, BaseExtractor] = {
            Task.VIS_PAS_ANALYSIS: MMRefExtractor(
                list(self.cases),
                exophora_referent_types,
            ),
            Task.VIS_COREFERENCE_RESOLUTION: MMRefExtractor(
                ["="], exophora_referent_types
            ),
        }
        self.task_to_rels: dict[Task, list[str]] = {
            Task.VIS_PAS_ANALYSIS: self.cases,
            Task.VIS_COREFERENCE_RESOLUTION: ["="],
        }

        # load visual annotations
        visual_annotation: list[ImageTextAnnotation] = self._load_visual_annotation(
            self.data_path, "json"
        )

        # visual annotations tailored for documents
        self.doc_id2vis: dict[str, dict] = {}
        for document in self.documents:
            # sub_doc_id -> orig_doc_id
            orig_doc_id: str = to_orig_doc_id(document.doc_id)
            doc_sentence_indices = [sentence.sid for sentence in document.sentences]
            base_phrases_to_vis: list[dict] = []
            utterances = visual_annotation[orig_doc_id].utterances
            for utterance in utterances:
                if utterance.sid not in doc_sentence_indices:
                    continue
                base_phrases_to_vis.extend(utterance.phrases)
            self.doc_id2vis.update({document.doc_id: base_phrases_to_vis})

        self.examples: list[MMRefExample] = self._load_examples(
            self.documents, str(data_path)
        )

        self.special_encoding: Encoding = self.tokenizer(
            self.special_tokens,
            is_split_into_words=True,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=False,
            add_special_tokens=False,
        ).encodings[0]

    @staticmethod
    def _load_visual_annotation(data_path: Path, ext: str = "json") -> dict[str, dict]:
        visuals = {}
        assert data_path.is_dir()
        for path in sorted(data_path.glob(f"*.{ext}")):
            annot = json.load(open(path, "r", encoding="utf-8"))
            annot = ImageTextAnnotation(**annot)  # for faster loading
            visuals.update({annot.scenarioId: annot})
        return visuals

    @staticmethod
    def _load_object(data_path: Path, file_id: str, ext: str = "pth") -> list[dict]:
        assert data_path.is_dir()
        path = data_path / (file_id + f".{ext}")
        return torch.load(path)

    @property
    def special_indices(self) -> list[int]:
        return list(self.special_to_index.values())

    @property
    def num_special_tokens(self) -> int:
        return len(self.special_tokens)

    @property
    def rel_types(self) -> list[str]:
        return [rel_type for task in self.tasks for rel_type in self.task_to_rels[task]]

    def _load_examples(
        self, documents: list[Document], documents_path: str
    ) -> list[MMRefExample]:
        """Loads examples from knp document and visual annotation"""
        examples = []
        load_cache: bool = (
            "DISABLE_CACHE" not in os.environ and "OVERWRITE_CACHE" not in os.environ
        )
        save_cache: bool = "DISABLE_CACHE" not in os.environ
        mmref_cache_dir: Path = Path(
            os.environ.get("CACHE_DIR", f'/tmp/{os.environ["USER"]}/mmref_cache')
        )
        for document in tqdm(documents, desc="Loading examples"):
            hash_ = self._hash(
                documents_path,
                self.doc_id2vis,
                self.tasks,
                self.task_to_extractor,
                self.flip_reader_writer,
            )
            example_cache_path = mmref_cache_dir / hash_ / f"{document.doc_id}.pkl"
            if example_cache_path.exists() and load_cache:
                with example_cache_path.open(mode="rb") as f:
                    try:
                        example = pickle.load(f)
                    except EOFError:
                        example = self._load_example_from_document(document)
            else:
                example = self._load_example_from_document(document)
                if save_cache:
                    self._save_cache(example, example_cache_path)
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
        self, examples: list[MMRefExample]
    ) -> list[MMRefExample]:
        idx = 0
        filtered = []
        for example in examples:
            phrases = next(iter(example.phrases.values()))

            # truncate or pad candidates
            for phrase in phrases:
                if phrase.rel2tags is not None:
                    if len(phrase.referent_candidates) > self.max_seq_length:
                        phrase = self._truncate_candidates(phrase, self.max_seq_length)
                    else:
                        phrase = self._pad_candidates(phrase, self.max_seq_length)

            encoding: Encoding = self.tokenizer(
                " ".join(
                    [morpheme for phrase in phrases for morpheme in phrase.morphemes]
                ),
                is_split_into_words=False,
                padding=PaddingStrategy.MAX_LENGTH,
                truncation=False,
                max_length=self.max_seq_length - self.num_special_tokens,
            ).encodings[0]
            if len(encoding.ids) > self.max_seq_length - self.num_special_tokens:
                continue
            example.encoding = encoding
            example.example_id = idx
            filtered.append(example)
            idx += 1
        return filtered

    @staticmethod
    def _truncate_candidates(phrase: MMRefBasePhrase, max_seq_length: int) -> MMRefBasePhrase:
        phrase.referent_candidates = phrase.referent_candidates[:max_seq_length]
        filtered_rel2tags: dict[str, list[int]] = {}
        for rel in phrase.rel2tags:
            filtered_rel2tags[rel] = [idx for idx in phrase.rel2tags[rel] if idx < max_seq_length]
        phrase.rel2tags = filtered_rel2tags
        return phrase

    @staticmethod
    def _pad_candidates(phrase: MMRefBasePhrase, max_seq_length: int) -> MMRefBasePhrase:
        assert len(phrase.referent_candidates) > 0
        emb_size: torch.Size = phrase.referent_candidates[0].feature.shape
        pad_mask: ObjectFeature = ObjectFeature(
            class_id=torch.Tensor([-1.0]),
            score=torch.Tensor([0.0]),
            bbox=torch.zeros(4),
            feature=torch.zeros(emb_size)
        )
        phrase.referent_candidates += [pad_mask] * (max_seq_length - len(phrase.referent_candidates))
        return phrase

    def _load_example_from_document(self, document: Document) -> MMRefExample:
        visual_phrases: dict = self.doc_id2vis[document.doc_id]
        orig_doc_id: str = to_orig_doc_id(document.doc_id)
        sid_to_objects: dict[str, list] = {sent.sid: [] for sent in document.sentences}

        if self.is_jcre3_dataset:
            jcre3_dataset_dir = self.dataset_path / "jcre3" / "recording"
            dataset_info = DatasetInfo.from_json(
                (jcre3_dataset_dir / orig_doc_id / "info.json").read_text()
            )
            obj_features = self._load_object(self.data_path, orig_doc_id, ext="pth")

            for utterance in dataset_info.utterances:
                if len(utterance.image_ids) == 0:
                    continue
                # info.jsonに記載の発話区間 + (and -) "image_input_width" フレーム
                sidx = max(0, int(utterance.image_ids[0]) - 1 - self.image_input_width)
                eidx = min(
                    len(obj_features) - 1,
                    int(utterance.image_ids[-1]) + self.image_input_width,
                )
                assert eidx >= sidx
                sid_to_objects.update(
                    {sid: obj_features[sidx:eidx] for sid in utterance.sids}
                )

        example = MMRefExample()
        example.load(
            document=document,
            visual_phrases=visual_phrases,
            tasks=self.tasks,
            task_to_extractor=self.task_to_extractor,
            sid_to_objects=sid_to_objects,
        )
        return example

    # TODO:
    def _convert_example_to_feature(self, example: MMRefExample) -> MMRefInputFeatures:
        """Loads a data file into a list of input features"""
        pass

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> MMRefInputFeatures:
        return self._convert_example_to_feature(self.examples[idx])
