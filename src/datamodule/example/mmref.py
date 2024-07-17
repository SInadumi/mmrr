from typing import Optional

from rhoknp import BasePhrase, Document
from tokenizers import Encoding

from cohesion_tools.extractors.base import BaseExtractor
from cohesion_tools.task import Task
from utils.dataset import MMRefBasePhrase, ObjectFeature
from utils.sub_document import extract_target_sentences


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
                referent_candidates=[],
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
                rel2tags: dict[str, list[int]] = extractor.extract_rels(
                    visual_phrase, candidates
                )
                mmref_base_phrase.referent_candidates = candidates
                mmref_base_phrase.rel2tags= rel2tags
        return mmref_base_phrases

    @staticmethod
    def _get_object_candidates(objects: list[dict]) -> list[ObjectFeature]:
        """Get object candidates for parsing"""
        ret: list[ObjectFeature] = []
        for objs in objects:
            for idx, class_id in enumerate(objs["classes"]):
                ret.append(
                    ObjectFeature(
                        class_id=class_id,
                        score=objs["scores"][idx],
                        bbox=objs["boxes"][idx],
                        feature=objs["feats"][idx],
                    )
                )
        return ret
