import hashlib
import json
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import ListConfig
from rhoknp import Document, RegexSenter
from rhoknp.cohesion import ExophoraReferent, ExophoraReferentType
from tokenizers import Encoding
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy

from cohesion_tools.extractors import MMRefExtractor
from cohesion_tools.extractors.base import BaseExtractor
from cohesion_tools.task import Task
from datamodule.example.mmref import MMRefExample
from utils.sub_document import to_idx_from_sid, to_orig_doc_id
from utils.util import DatasetInfo

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextualFeatures:
    input_ids: list[int]
    attention_mask: list[bool]
    token_type_ids: list[int]
    source_mask: list[bool]  # loss を計算する対象の基本句かどうか
    source_label: list[list[int]]  # 解析対象基本句かどうか


@dataclass(frozen=True)
class VisualFeatures:
    input_embeds: list[torch.Tensor]
    attention_mask: list[bool]
    target_mask: list[
        list[list[bool]]
    ]  # source と関係を持つ候補かどうか（後ろと共参照はしないなど）
    target_label: list[list[list[float]]]  # source と関係を持つかどうか


@dataclass(frozen=True)
class InputFeatures:
    """A dataclass which represents a language encoder and interaction layer input

    TODO: The attributes of this class correspond to arguments of forward method of each encoder
    """

    example_id: int
    textual: TextualFeatures
    visual: VisualFeatures


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
            Task.VIS_COREFERENCE_RESOLUTION: MMRefExtractor(["="], exophora_referent_types),
        }
        self.task_to_rels: dict[Task, list[str]] = {
            Task.VIS_PAS_ANALYSIS: self.cases,
            Task.VIS_COREFERENCE_RESOLUTION: ["="],
        }

        # load visual annotations
        vis_grounding = self._load_visual_grounding(self.data_path, "json")
        iid_to_cid = self._get_instance_id_mapper(self.dataset_path, vis_grounding)

        # visual grounding annotations tailored for documents
        self.doc_id2grounding: dict[str, dict] = {}
        for document in self.documents:
            orig_doc_id: str = to_orig_doc_id(
                document.doc_id
            )  # sub_doc_id -> orig_doc_id
            mapper: dict[str, int] = iid_to_cid[orig_doc_id]
            utterances = self._split_utterances(
                document, vis_grounding[orig_doc_id]["utterances"]
            )
            base_phrases_to_vis: list[dict] = []
            for sentence in document.sentences:
                sid = to_idx_from_sid(sentence.sid)
                utt = utterances[sid]
                # find category id with mapper
                for p in utt["phrases"]:
                    for rel in p["relations"]:
                        #  HACK: visual/*.json の "images"中にclassNameが存在しないinstanceIdを回避する例外処理
                        try:
                            rel["categoryId"] = mapper[rel["instanceId"]]
                        except Exception as e:
                            rel["categoryId"] = -1  # HACK: dummy category ID
                            print(f"{orig_doc_id}: {e.__class__.__name__}: {e}")
                base_phrases_to_vis.extend(utt["phrases"])
            self.doc_id2grounding.update({document.doc_id: base_phrases_to_vis})

        self.examples: list[MMRefExample] = self._load_examples(
            self.documents, str(data_path)
        )

        self.special_encoding: Encoding = self.tokenizer(
            self.special_tokens,
            is_split_into_words=True,
            padding=PaddingStrategy.NO_NOT_PAD,
            truncation=False,
            add_special_tokens=False,
        ).encodings[0]

    @staticmethod
    def _load_visual_grounding(data_path: Path, ext: str = "json") -> dict[str, dict]:
        visuals = {}
        assert data_path.is_dir()
        for path in sorted(data_path.glob(f"*.{ext}")):
            vis = json.load(open(path, "r", encoding="utf-8"))
            visuals.update({vis["scenarioId"]: vis})
        return visuals

    @staticmethod
    def _load_object(data_path: Path, file_id: str, ext: str = "pth") -> list[dict]:
        assert data_path.is_dir()
        path = data_path / (file_id + f".{ext}")
        return torch.load(path)

    @staticmethod
    def _get_instance_id_mapper(
        dataset_path: Path, vis_grounding: dict
    ) -> dict[str, dict]:
        mappers = {}  # NOTE: key: orig_doc_id, value: a mapper from instanceId to className
        id2cat = json.load(
            open(dataset_path / "categories.json", "r", encoding="utf-8")
        )
        cat2id = {v: i for i, v in enumerate(id2cat)}

        for orig_doc_id, annot in vis_grounding.items():
            mapper: dict[str, str] = {}
            image_annot = annot["images"]
            for img in image_annot:
                for bbox in img["boundingBoxes"]:
                    mapper.update({bbox["instanceId"]: cat2id[bbox["className"]]})
            mappers.update({orig_doc_id: mapper})
        return mappers

    @staticmethod
    def _split_utterances(document: Document, utterances: list[dict]) -> list[dict]:
        """visual_annotation/*.jsonの"utterances"エントリをRegexSenterでtextual_annotation/*.knpの文分割に揃える処理

        c.f.) https://rhoknp.readthedocs.io/en/stable/_modules/rhoknp/processors/senter.html
        TODO: "info.json"を参照する実装に変更
        """
        # split utterances field
        senter = RegexSenter()
        ret = []
        for utt in utterances:
            sents = senter.apply_to_document(utt["text"])
            ret.extend(
                [{"text": s.text, "phrases": utt["phrases"]} for s in sents.sentences]
            )

        # format "ret" phrase entries by base_phrases
        for sentence in document.sentences:
            s_idx = to_idx_from_sid(sentence.sid)  # a index of sid
            utt = ret[s_idx]
            if len(sentence.base_phrases) != len(utt["phrases"]):
                doc_phrase = [b.text for b in sentence.base_phrases]
                vis_phrase = [up["text"] for up in utt["phrases"]]
                st_idx = vis_phrase.index(doc_phrase[0])
                end_idx = st_idx + len(doc_phrase)
                utt["phrases"] = utt["phrases"][st_idx:end_idx]
        return ret

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
                self.doc_id2grounding,
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

    def _load_example_from_document(self, document: Document) -> MMRefExample:
        visual_phrases: dict = self.doc_id2grounding[document.doc_id]
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
                # NOTE: info.jsonに記載の発話区間 + (and -) "image_input_width" フレーム
                st_idx = max(
                    0, int(utterance.image_ids[0]) - 1 - self.image_input_width
                )
                end_idx = min(
                    len(obj_features) - 1,
                    int(utterance.image_ids[-1]) + self.image_input_width,
                )
                assert end_idx >= st_idx
                sid_to_objects.update(
                    {sid: obj_features[st_idx:end_idx] for sid in utterance.sids}
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
    def _convert_example_to_feature(self, example: MMRefExample) -> InputFeatures:
        """Loads a data file into a list of input features"""
        pass

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> InputFeatures:
        return self._convert_example_to_feature(self.examples[idx])
