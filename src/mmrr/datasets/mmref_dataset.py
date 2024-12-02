import hashlib
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import ListConfig
from rhoknp import Document, Sentence
from rhoknp.cohesion import ExophoraReferent, ExophoraReferentType
from tokenizers import Encoding
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy

from mmrr.datamodule.example import MMRefExample
from mmrr.tools.extractors import MMRefExtractor
from mmrr.tools.extractors.base import BaseExtractor
from mmrr.tools.task import Task
from mmrr.utils.annotation import ImageTextAnnotation, SentenceAnnotation
from mmrr.utils.dataset import (
    MMRefBasePhrase,
    MMRefInputFeatures,
)
from mmrr.utils.prediction import ObjectFeature
from mmrr.utils.util import IGNORE_ID, Rectangle, sigmoid

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
        object_file_root: Union[str, Path],
        object_file_name: str,
        object_hidden_size: int,
        tokenizer: PreTrainedTokenizerBase,
        exophora_referents: ListConfig,
        include_nonidentical: bool,
        special_tokens: ListConfig,
        training: bool,
    ) -> None:
        super().__init__(
            tasks=[Task(task) for task in tasks],
            tokenizer=tokenizer,
            training=training,
        )

        self.dataset_path = Path(dataset_path)
        self.data_path = Path(data_path)
        self.exophora_referents: list[ExophoraReferent] = [
            ExophoraReferent(s) for s in exophora_referents
        ]
        self.special_tokens: list[str] = list(special_tokens)
        self.cases = list(cases)
        self.max_seq_length = max_seq_length
        self.object_hidden_size = object_hidden_size
        self.dataset_name = self.data_path.parts[-2]

        assert len(self.tasks) > 0
        assert self.max_seq_length > 0
        assert self.object_hidden_size > 0

        exophora_referent_types: list[ExophoraReferentType] = [
            er.type for er in self.exophora_referents
        ]
        self.task_to_extractor: dict[Task, BaseExtractor] = {
            Task.MM_PAS_ANALYSIS: MMRefExtractor(
                self.cases,
                exophora_referent_types,
                include_nonidentical,
            ),
            Task.MM_COREFERENCE_RESOLUTION: MMRefExtractor(
                ["="],
                exophora_referent_types,
                include_nonidentical,
            ),
        }
        self.task_to_rels: dict[Task, list[str]] = {
            Task.MM_PAS_ANALYSIS: self.cases,
            Task.MM_COREFERENCE_RESOLUTION: ["="],
        }

        # load visual annotations
        image_text_annotations: list[ImageTextAnnotation] = (
            self._load_visual_annotation(self.data_path, "json")
        )
        self.sid2vis_sentence: dict[str, SentenceAnnotation] = {}
        for annotation in image_text_annotations:
            for utterance in annotation.utterances:
                assert isinstance(utterance.sid, str)
                self.sid2vis_sentence.update({utterance.sid: utterance})

        # load textual annotations
        documents: list[Document] = self.load_documents(self.data_path)
        sid2knp_sentence: dict[str, Sentence] = {}
        for document in documents:
            for sentence in document.sentences:
                sid2knp_sentence.update({sentence.sid: sentence})

        # load object features
        self.object_file_name = object_file_name
        self.objects = h5py.File(Path(object_file_root) / f"{object_file_name}.h5", "r")
        self.iou_mapper = h5py.File(
            Path(object_file_root) / f"{object_file_name}_iou_mapper.h5", "r"
        )

        self.pad_mask = ObjectFeature(feature=torch.zeros(self.object_hidden_size))
        self.special_encoding: Encoding = self.tokenizer(
            self.special_tokens,
            is_split_into_words=True,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=False,
            add_special_tokens=False,
        ).encodings[0]

        try:
            self.examples: list[MMRefExample] = self._load_examples_per_frame(
                image_text_annotations, sid2knp_sentence
            )
        except Exception as e:
            logger.error(f"{type(e).__name__}: {e}")
            sys.exit(1)
        finally:
            self.objects.close()
            self.iou_mapper.close()

    @staticmethod
    def _load_visual_annotation(
        data_path: Path, ext: str = "json"
    ) -> list[ImageTextAnnotation]:
        ret: list[ImageTextAnnotation] = []
        assert data_path.is_dir()
        for path in sorted(data_path.glob(f"*.{ext}")):
            raw_annot = json.load(
                open(path, "r", encoding="utf-8")
            )  # for faster loading
            ret.append(ImageTextAnnotation(**raw_annot))
        return ret

    def _load_objects(self, scenario_id: str, image_id: str) -> list[ObjectFeature]:
        ret: list[ObjectFeature] = []
        boxes = list(self.objects[f"{scenario_id}/{image_id}/boxes"])
        classes = list(self.objects[f"{scenario_id}/{image_id}/classes"])
        feats = list(self.objects[f"{scenario_id}/{image_id}/feats"])
        scores = list(self.objects[f"{scenario_id}/{image_id}/scores"])
        for idx in range(len(boxes)):
            _bbox = boxes[idx].tolist()
            ret.append(
                ObjectFeature(
                    image_id=int(image_id),
                    class_id=int(classes[idx]),
                    confidence=float(scores[idx]),
                    rect=Rectangle(x1=_bbox[0], y1=_bbox[1], x2=_bbox[2], y2=_bbox[3]),
                    feature=torch.Tensor(feats[idx]),
                )
            )
        return ret

    @property
    def rel_types(self) -> list[str]:
        return [rel_type for task in self.tasks for rel_type in self.task_to_rels[task]]

    def _load_examples_per_frame(
        self,
        image_text_annotations: list[ImageTextAnnotation],
        sid2knp_sentence: dict[str, Sentence],
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
        hash_ = self._hash(
            self.data_path,
            self.tasks,
            self.task_to_extractor,
            self.object_file_name,
        )
        for idx, annotation in enumerate(
            tqdm(
                image_text_annotations,
                desc="Loading examples",
                dynamic_ncols=True,
            )
        ):
            assert len(annotation.images) == 1, "single images/frames only"
            image_id: str = annotation.images[0].imageId
            scenario_id = annotation.scenarioId
            knp_document = Document.from_sentences(
                [sid2knp_sentence[utt.sid] for utt in annotation.utterances]
            )
            candidates: list[ObjectFeature] = self._load_objects(
                scenario_id, image_id
            )  # Loading object candidates
            iou_mapper: dict[str, h5py.Group] = {
                bbox.instanceId: self.iou_mapper[
                    f"{scenario_id}/{image_id}/{bbox.instanceId}"
                ]
                for bbox in annotation.images[0].boundingBoxes
            }
            example_cache_path = mmref_cache_dir / hash_ / f"{scenario_id}-{idx}.pkl"
            if example_cache_path.exists() and load_cache:
                with example_cache_path.open(mode="rb") as f:
                    try:
                        example = pickle.load(f)
                    except EOFError:
                        example = MMRefExample()
                        example = example.load(
                            document=knp_document,
                            tasks=self.tasks,
                            task_to_extractor=self.task_to_extractor,
                            image_id=image_id,
                            vis_sentences=annotation.utterances,
                            candidates=candidates,
                            iou_mapper=iou_mapper,
                        )
            else:
                example = MMRefExample()
                example.load(
                    document=knp_document,
                    tasks=self.tasks,
                    task_to_extractor=self.task_to_extractor,
                    image_id=image_id,
                    vis_sentences=annotation.utterances,
                    candidates=candidates,
                    iou_mapper=iou_mapper,
                )
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
        self,
        examples: list[MMRefExample],
    ) -> list[MMRefExample]:
        filtered = []
        for idx, example in enumerate(examples):
            phrases = next(iter(example.phrases.values()))
            morphemes = [
                morpheme for phrase in phrases for morpheme in phrase.morphemes
            ]
            encoding: Encoding = self.tokenizer(
                " ".join(morphemes),
                is_split_into_words=False,
                padding=PaddingStrategy.MAX_LENGTH,
                truncation=False,
                max_length=self.max_seq_length - len(self.special_tokens),
            ).encodings[0]
            if len(encoding.ids) > self.max_seq_length - len(self.special_tokens):
                continue
            example.encoding = encoding
            example.example_id = idx

            # truncate candidates
            if len(example.candidates) > self.max_seq_length:
                example.candidates, example.phrases = self._truncate_candidates(
                    example.candidates,
                    example.phrases,
                )
            # pad candidates
            if len(example.candidates) < self.max_seq_length:
                example.candidates = self._pad_candidates(example.candidates)

            filtered.append(example)
        return filtered

    def _truncate_candidates(
        self,
        candidates: list[ObjectFeature],
        phrases: dict[Task, list[MMRefBasePhrase]],
    ) -> tuple[list[ObjectFeature], dict[Task, list[MMRefBasePhrase]]]:
        max_seq_length = self.max_seq_length
        candidates = candidates[:max_seq_length]
        for _, task_phrases in phrases.items():
            for task_phrase in task_phrases:
                if task_phrase.rel2tags is None:
                    continue
                for rel, tags in task_phrase.rel2tags.items():
                    task_phrase.rel2tags[rel] = [
                        tag for tag in tags if tag < max_seq_length
                    ]
        return candidates, phrases

    def _pad_candidates(
        self,
        candidates: list[ObjectFeature],
    ) -> list[ObjectFeature]:
        max_seq_length = self.max_seq_length
        candidates += [self.pad_mask] * (max_seq_length - len(candidates))
        return candidates

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
                    example.candidates,
                    example.encoding,
                )
            )
        return np.array(predictions).transpose(1, 0, 2)  # (phrase, rel, candidate)

    def _token_to_candidate_level(
        self,
        token_level_logits_matrix: np.ndarray,  # (seq, seq)
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
            ].tolist()  # (seq)
            candidate_level_logits: list[float] = []
            for idx, candidate in enumerate(candidates):
                if candidate.image_id == IGNORE_ID or candidate.class_id == IGNORE_ID:
                    continue
                candidate_level_logits.append(token_level_logits[idx])
            phrase_level_scores_matrix.append(sigmoid(np.array(candidate_level_logits)))
        return np.array(phrase_level_scores_matrix)

    def _convert_example_to_feature(
        self,
        example: MMRefExample,
    ) -> MMRefInputFeatures:
        """Convert example to textual feature"""
        merged_encoding: Encoding = Encoding.merge(
            [example.encoding, self.special_encoding]
        )

        """Convert example to visual feature"""
        vis_embeds_list: list[torch.Tensor] = []
        vis_attention_mask: list[bool] = []
        for candidate in example.candidates:
            vis_embeds_list.append(candidate.feature)
            vis_attention_mask.append(True if candidate.class_id != -1 else False)
        vis_embeds = torch.stack(vis_embeds_list)  # -> torch.Tensor

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
            [0.0] * self.max_seq_length for _ in range(self.max_seq_length)
        ]  # (src, tgt)
        candidates_set: list[list[bool]] = [
            [False] * self.max_seq_length for _ in range(self.max_seq_length)
        ]  # (src, tgt)

        for phrase in phrases:
            scores: list[float] = [0.0] * self.max_seq_length
            token_level_candidates: list[bool] = [False] * self.max_seq_length
            # phrase.rel2tags が None の場合は推論時，もしくは学習対象外の物体候補．
            # その場合は scores が全てゼロになるため loss が計算されない．
            if phrase.rel2tags is not None:
                # 学習・解析対象物体
                for cid in phrase.rel2tags.get(rel_type, []):
                    assert cid < self.max_seq_length
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
