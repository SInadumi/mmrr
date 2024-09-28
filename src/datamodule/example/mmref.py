from rhoknp import BasePhrase, Document

from cohesion_tools.extractors.base import BaseExtractor
from cohesion_tools.task import Task
from utils.annotation import PhraseAnnotation
from utils.dataset import MMRefBasePhrase
from utils.prediction import ObjectFeature

from .base import BaseExample


class MMRefExample(BaseExample):
    def __init__(self) -> None:
        self.phrases: dict[Task, list[MMRefBasePhrase]] = {}
        self.sid_to_objects: dict[str, list] = {}
        self.all_candidates: list[ObjectFeature] = None

    def load(
        self,
        document: Document,
        visual_phrases: dict[list],
        tasks: list[Task],
        task_to_extractor: dict[Task, BaseExtractor],
        sid_to_objects: dict[str, list],
    ):
        self.set_doc_params(document)
        self.sid_to_objects = sid_to_objects
        for task in tasks:
            extractor: BaseExtractor = task_to_extractor[task]
            self.phrases[task] = self._wrap_base_phrases(
                document.base_phrases,
                visual_phrases,
                extractor,
            )

    def _wrap_base_phrases(
        self,
        base_phrases: list[BasePhrase],
        visual_phrases: list[PhraseAnnotation],
        extractor: BaseExtractor,
    ) -> list[MMRefBasePhrase]:
        mmref_base_phrases = [
            MMRefBasePhrase(
                base_phrase.head.global_index,
                [morpheme.global_index for morpheme in base_phrase.morphemes],
                [morpheme.text for morpheme in base_phrase.morphemes],
                is_target=False,
                referent_candidates=[],
            )
            for base_phrase in base_phrases
        ]

        assert len(base_phrases) == len(mmref_base_phrases) == len(visual_phrases)
        for idx in range(len(base_phrases)):
            base_phrase: BasePhrase = base_phrases[idx]
            visual_phrase: PhraseAnnotation = visual_phrases[idx]
            mmref_base_phrase: MMRefBasePhrase = mmref_base_phrases[idx]

            # set a parameter: "is_target"
            objects = self.sid_to_objects[base_phrase.sentence.sid]
            mmref_base_phrase.is_target = extractor.is_target(visual_phrase)

            # set parameters: "referent_candidates" and "rel2tags"
            if mmref_base_phrase.is_target:
                candidates: list[ObjectFeature] = self._get_object_candidates(objects)
                rel2tags: dict[str, list[int]] = extractor.extract_rels(
                    visual_phrase, candidates
                )
                mmref_base_phrase.referent_candidates = candidates
                mmref_base_phrase.rel2tags = rel2tags
        return mmref_base_phrases

    @staticmethod
    def _get_object_candidates(objects: list[dict]) -> list[ObjectFeature]:
        """Get object candidates for parsing"""
        ret: list[ObjectFeature] = []
        for objs in objects:
            for idx, class_id in enumerate(objs["classes"]):
                ret.append(
                    ObjectFeature(
                        image_id=objs["image_id"],
                        class_id=class_id,
                        score=objs["scores"][idx],
                        bbox=objs["boxes"][idx],
                        feature=objs["feats"][idx],
                    )
                )
        # sort object candidates by detector confidences
        return sorted(ret, key=lambda x: x.score.item(), reverse=True)
