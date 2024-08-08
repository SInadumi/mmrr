import hashlib
import json
import logging
import os
import pickle
import random
from pathlib import Path
from typing import Union

import numpy as np
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
from datamodule.example import MMRefExample
from utils.annotation import DatasetInfo, ImageTextAnnotation, PhraseAnnotation
from utils.dataset import (
    MMRefBasePhrase,
    MMRefInputFeatures,
    ObjectFeature,
)
from utils.sub_document import to_orig_doc_id
from utils.util import IGNORE_INDEX, sigmoid

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
        vis_max_seq_length: int,
        vis_emb_size: int,
        document_split_stride: int,
        tokenizer: PreTrainedTokenizerBase,
        exophora_referents: ListConfig,
        special_tokens: ListConfig,
        training: bool,
        flip_reader_writer: bool,
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
        self.vis_max_seq_length = (
            vis_max_seq_length - 1
        )  # -1: for adding "no object" feature
        self.vis_emb_size = vis_emb_size
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
        self.doc_id2vis: dict[str, list[PhraseAnnotation]] = {}
        for document in self.documents:
            # sub_doc_id -> orig_doc_id
            orig_doc_id = to_orig_doc_id(document.doc_id)
            doc_sentence_indices = [sentence.sid for sentence in document.sentences]
            base_phrases_to_vis: list[PhraseAnnotation] = []
            utterances = visual_annotation[orig_doc_id].utterances
            for utterance in utterances:
                if utterance.sid not in doc_sentence_indices:
                    continue
                base_phrases_to_vis.extend(utterance.phrases)
            self.doc_id2vis.update({document.doc_id: base_phrases_to_vis})

        self.no_object_feature: ObjectFeature = ObjectFeature(
            class_id=torch.Tensor([0.0]),
            score=torch.Tensor([0.0]),
            bbox=torch.zeros(4),
            feature=torch.ones(self.vis_emb_size),
        )

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
        visuals: dict[str, ImageTextAnnotation] = {}
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

            # collect candidates
            all_candidates, phrases = self._collect_candidates(phrases)

            # truncate or pad candidates
            if len(all_candidates) > self.vis_max_seq_length:
                all_candidates, phrases = self._truncate_candidates(
                    all_candidates, phrases
                )
            else:
                all_candidates = self._pad_candidates(all_candidates)

            # add "no object" feature to the last element of list
            all_candidates.append(self.no_object_feature)

            example.all_candidates = all_candidates

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
    def _collect_candidates(
        phrases: list[MMRefBasePhrase],
    ) -> tuple[list[ObjectFeature], list[MMRefBasePhrase]]:
        all_candidates: list[ObjectFeature] = []
        candidate_sidx: list[int] = [0]
        for idx, phrase in enumerate(phrases):
            all_candidates.extend(phrase.referent_candidates)
            candidate_sidx.append(candidate_sidx[idx] + len(phrase.referent_candidates))

        for idx, phrase in enumerate(phrases):
            if phrase.rel2tags is not None:
                sidx = candidate_sidx[idx]
                for rel, cids in phrase.rel2tags.items():
                    phrase.rel2tags[rel] = [cid + sidx for cid in cids]
        return all_candidates, phrases

    def _truncate_candidates(
        self,
        all_candidates: list[ObjectFeature],
        phrases: list[MMRefBasePhrase],
    ) -> tuple[list[ObjectFeature], list[MMRefBasePhrase]]:
        max_seq_length: int = self.vis_max_seq_length

        # collect pos/neg candidate indices
        pos_cand_indices = set()
        for phrase in phrases:
            if phrase.rel2tags is not None:
                pos_cand_indices |= set(
                    [cid for cids in phrase.rel2tags.values() for cid in cids]
                )
        neg_cand_indices = set(range(len(all_candidates))) - set(pos_cand_indices)

        # truncate negative candidates
        assert max_seq_length - len(pos_cand_indices) > 0
        pos_cand_indices = list(pos_cand_indices)
        neg_cand_indices = list(neg_cand_indices)
        neg_cand_indices = random.sample(
            neg_cand_indices, max_seq_length - len(pos_cand_indices)
        )
        all_cand_indices = sorted(pos_cand_indices + neg_cand_indices)
        all_candidates = [all_candidates[idx] for idx in all_cand_indices]

        # reallocate rel2tag values
        pos_mapper: dict[int, int] = {
            pidx: nidx
            for nidx, pidx in enumerate(all_cand_indices)
            if pidx in pos_cand_indices
        }  # prev idx to new idx
        for phrase in phrases:
            if phrase.rel2tags is not None:
                for rel, cids in phrase.rel2tags.items():
                    phrase.rel2tags[rel] = [pos_mapper[cid] for cid in cids]

        return all_candidates, phrases

    def _pad_candidates(
        self,
        all_candidates: list[ObjectFeature],
    ) -> MMRefBasePhrase:
        emb_size: torch.Size = self.vis_emb_size
        max_seq_length: int = self.vis_max_seq_length
        pad_mask: ObjectFeature = ObjectFeature(
            class_id=torch.Tensor([-1.0]),
            score=torch.Tensor([0.0]),
            bbox=torch.zeros(4),
            feature=torch.zeros(emb_size),
        )
        all_candidates += [pad_mask] * (max_seq_length - len(all_candidates))
        return all_candidates

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
                sidx, eidx = utterance.image_indices_span
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

    def dump_relation_prediction(
        self,
        relation_logits: np.ndarray,
        example: MMRefExample,
    ) -> np.ndarray:
        """1 example 中に存在する基本句それぞれに対してシステム予測のリストを返す．"""
        predictions: list[np.ndarray] = []
        task_and_rels = [
            (task, rel) for task in self.tasks for rel in self.task_to_rels[task]
        ]
        assert len(relation_logits) == len(task_and_rels) == len(self.rel_types)
        for (task, _), logits in zip(task_and_rels, relation_logits):
            predictions.append(
                self._token_to_candidate_level(
                    logits,
                    example.phrases[task],
                    example.all_candidates,
                    example.encoding,
                )
            )
        return np.array(predictions).transpose(1, 0, 2)  # (phrase, rel, candidate)

    def _token_to_candidate_level(
        self,
        token_level_logits_matrix: np.ndarray,  # (t_seq, v_seq)
        phrases: list[MMRefBasePhrase],
        candidates: list[ObjectFeature],
        encoding: Encoding,
    ) -> np.ndarray:  # (phrase, candidate)
        phrase_level_scores_matrix: list[np.ndarray] = []
        for phrase in phrases:
            token_index_span: tuple[int, int] = encoding.word_to_tokens(
                phrase.head_morpheme_global_index
            )
            # Use the head subword as the representative of the source word.
            # Cast to built-in list because list operation is faster than numpy array operation.
            token_level_logits: list[float] = token_level_logits_matrix[
                token_index_span[0]
            ].tolist()  # (t_seq)
            candidate_level_logits: list[float] = []
            for idx in range(len(candidates)):
                candidate_level_logits.append(token_level_logits[idx])
            phrase_level_scores_matrix.append(sigmoid(np.array(candidate_level_logits)))
        return np.array(phrase_level_scores_matrix)

    def _convert_example_to_feature(
        self,
        example: MMRefExample,
    ) -> MMRefInputFeatures:
        """Convert example to textual feature"""
        assert example.encoding is not None, "encoding isn't set"
        source_mask = [False] * self.max_seq_length
        for global_index in example.analysis_target_morpheme_indices:
            for token_index in range(*example.encoding.word_to_tokens(global_index)):
                source_mask[token_index] = True

        is_analysis_targets: list[list[int]] = []  # (task, src)
        for task in self.tasks:
            is_targets: list[int] = [IGNORE_INDEX] * self.max_seq_length
            for phrase in example.phrases[task]:
                token_index_span: tuple[int, int] = example.encoding.word_to_tokens(
                    phrase.head_morpheme_global_index
                )
                for token_index in range(*token_index_span):
                    is_targets[token_index] = int(phrase.is_target)
            is_analysis_targets.append(is_targets)

        merged_encoding: Encoding = Encoding.merge(
            [example.encoding, self.special_encoding]
        )

        """Convert example to visual feature"""
        vis_embeds: list[torch.Tensor] = []
        vis_attention_mask: list[bool] = []
        for candidate in example.all_candidates:
            vis_embeds.append(candidate.feature)
            vis_attention_mask.append(True if candidate.class_id != -1 else False)
        vis_embeds = torch.stack(vis_embeds)  # -> torch.Tensor

        scores_set: list[list[list[float]]] = []  # (rel, src, tgt)
        candidates_set: list[list[list[bool]]] = []  # (rel, src, tgt)
        for task in self.tasks:
            for rel in self.task_to_rels[task]:
                scores, candidates = self._convert_annotation_to_feature(
                    example.phrases[task], rel, example.encoding
                )
                scores_set.append(scores)
                candidates_set.append(candidates)

        return MMRefInputFeatures(
            example_id=example.example_id,
            input_ids=merged_encoding.ids,
            attention_mask=merged_encoding.attention_mask,
            token_type_ids=merged_encoding.type_ids,
            source_mask=source_mask,
            source_label=is_analysis_targets,
            vis_embeds=vis_embeds,
            vis_attention_mask=vis_attention_mask,
            target_mask=candidates_set,
            target_label=scores_set,
        )

    def _convert_annotation_to_feature(
        self,
        phrases: list[MMRefBasePhrase],
        rel_type: str,
        encoding: Encoding,
    ) -> tuple[list[list[float]], list[list[bool]]]:
        scores_set: list[list[float]] = [
            [0.0] * self.vis_max_seq_length for _ in range(self.max_seq_length)
        ]  # (src, tgt)
        candidates_set: list[list[bool]] = [
            [False] * self.vis_max_seq_length for _ in range(self.max_seq_length)
        ]  # (src, tgt)

        for phrase in phrases:
            scores: list[float] = [0.0] * self.vis_max_seq_length
            token_level_candidates: list[bool] = [False] * self.vis_max_seq_length
            # phrase.rel2tags が None の場合は推論時，もしくは学習対象外の物体候補．
            # その場合は scores が全てゼロになるため loss が計算されない．
            if phrase.rel2tags is not None:
                # 学習・解析対象物体
                for cid in phrase.rel2tags[rel_type]:
                    scores[cid] = 1.0
                    token_level_candidates[cid] = True

            token_index_span = encoding.word_to_tokens(
                phrase.head_morpheme_global_index
            )
            # use the head subword as the representative of the source word
            scores_set[token_index_span[0]] = scores
            candidates_set[token_index_span[0]] = token_level_candidates

        return scores_set, candidates_set

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> MMRefInputFeatures:
        return self._convert_example_to_feature(self.examples[idx])
