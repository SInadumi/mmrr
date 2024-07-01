from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import torch
from rhoknp import BasePhrase, Document
from tokenizers import Encoding

from cohesion_tools.extractors.base import BaseExtractor
from cohesion_tools.task import Task
from utils.sub_document import extract_target_sentences


@dataclass
class ObjectFeature:
    class_id: torch.Tensor
    score: torch.Tensor
    bbox: torch.Tensor
    feature: torch.Tensor


@dataclass
class MMRefBasePhrase:
    head_morpheme_global_index: int  # a phrase head index
    morpheme_global_indices: list[int]  # indices of phrase span
    morphemes: list[str]  # phrase span
    is_target: bool  # a flag of phrase span an analysis target
    positive_candidates: list["ObjectFeature"]
    negative_candidates: list["ObjectFeature"]


class MMRefExample:
    def __init__(self) -> None:
        self.example_id: int = -1
        self.doc_id: str = ""
        self.phrases: dict[Task, list[MMRefBasePhrase]] = {}
        self.sid_to_objects: dict[str, list] = {}
        self.analysis_target_morpheme_indices: list[int] = []
        self.encoding: Optional[Encoding] = None

    def load(
        self,
        document: Document,
        visual_phrases: dict[list],
        tasks: list[Task],
        task_to_extractor: dict[Task, BaseExtractor],
        sid_to_objects: dict[str, list],
    ):
        self.doc_id = document.doc_id
        self.sid_to_objects = sid_to_objects

        for task in tasks:
            extractor: BaseExtractor = task_to_extractor[task]
            self.phrases[task] = self._wrap_base_phrases(
                document.base_phrases,
                visual_phrases,
                extractor,
            )
        analysis_target_morpheme_indices = []
        for sentence in extract_target_sentences(document):
            analysis_target_morpheme_indices += [
                m.global_index for m in sentence.morphemes
            ]
        self.analysis_target_morpheme_indices = analysis_target_morpheme_indices

    def _wrap_base_phrases(
        self,
        base_phrases: list[BasePhrase],
        visual_phrases: list[dict[str, list]],
        extractor: BaseExtractor,
    ) -> list[MMRefBasePhrase]:
        mmref_base_phrases = [
            MMRefBasePhrase(
                base_phrase.head.global_index,
                [morpheme.global_index for morpheme in base_phrase.morphemes],
                [morpheme.text for morpheme in base_phrase.morphemes],
                is_target=extractor.is_target(visual_phrase),
                positive_candidates=[],
                negative_candidates=[],
            )
            for base_phrase, visual_phrase in zip(base_phrases, visual_phrases)
        ]
        assert len(base_phrases) == len(mmref_base_phrases) == len(visual_phrases)
        for idx in range(len(base_phrases)):
            # input
            base_phrase: BasePhrase = base_phrases[idx]
            visual_phrase: dict[str, list] = visual_phrases[idx]
            # output
            mmref_base_phrase: MMRefBasePhrase = mmref_base_phrases[idx]

            if mmref_base_phrase.is_target:
                candidates = self._get_object_candidates(
                    self.sid_to_objects[base_phrase.sentence.sid]
                )
                pos_candidates: dict[str, list] = extractor.extract_rels(
                    visual_phrase, candidates
                )
                neg_candidates: dict[str, list] = extractor.extract_rels(
                    visual_phrase, candidates, is_neg=True
                )
                # assert list(pos_candidates.values()) != [[[]]]
                mmref_base_phrase.positive_candidates = pos_candidates
                mmref_base_phrase.negative_candidates = neg_candidates
        return mmref_base_phrases

    @staticmethod
    def _get_object_candidates(objects: list[dict]) -> dict[int, list[ObjectFeature]]:
        """解析対象物体の候補を返す関数"""
        ret: dict[int, list[ObjectFeature]] = defaultdict(list)
        for objs in objects:
            for idx, cls_ in enumerate(objs["classes"]):
                ret[cls_.item()].append(
                    ObjectFeature(
                        class_id=cls_,
                        score=objs["scores"][idx],
                        bbox=objs["boxes"][idx],
                        feature=objs["feats"][idx],
                    )
                )
        # 物体クラス毎に最もconfidenceが高い物体候補を集計
        for k, v in ret.items():
            ret[k] = sorted(v, key=lambda x: x.score.item(), reverse=True)
        return ret
