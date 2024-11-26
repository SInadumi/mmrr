import h5py
from rhoknp import BasePhrase, Document

from mmrr.tools.extractors.base import BaseExtractor
from mmrr.tools.task import Task
from mmrr.utils.annotation import PhraseAnnotation, SentenceAnnotation
from mmrr.utils.dataset import MMRefBasePhrase
from mmrr.utils.prediction import ObjectFeature

from .base import BaseExample


class MMRefExample(BaseExample):
    def __init__(self) -> None:
        super().__init__()
        self.image_id: str = ""
        self.sentence_indices: list = []
        self.phrases: dict[Task, list[MMRefBasePhrase]] = {}
        self.sid_to_objects: dict[str, list] = {}
        self.candidates: list[ObjectFeature] = []

    def load(
        self,
        document: Document,
        tasks: list[Task],
        task_to_extractor: dict[Task, BaseExtractor],
        image_id: str,
        vis_sentences: list[SentenceAnnotation],
        candidates: list[ObjectFeature],
        iou_mapper: dict[str, h5py.Group],
    ):
        self.set_knp_params(document)
        self.image_id = image_id
        self.sentence_indices = [sentence.sid for sentence in vis_sentences]
        base_phrases: list[BasePhrase] = document.base_phrases
        vis_phrases: list[PhraseAnnotation] = [
            phrase for sentence in vis_sentences for phrase in sentence.phrases
        ]
        self.candidates = candidates
        assert len(base_phrases) == len(vis_phrases)
        for task in tasks:
            extractor: BaseExtractor = task_to_extractor[task]
            self.phrases[task] = self._wrap_base_phrases(
                base_phrases, vis_phrases, extractor, iou_mapper
            )

    def _wrap_base_phrases(
        self,
        base_phrases: list[BasePhrase],
        visual_phrases: list[PhraseAnnotation],
        extractor: BaseExtractor,
        iou_mapper: dict[str, h5py.Group],
    ) -> list[MMRefBasePhrase]:
        mmref_base_phrases = [
            MMRefBasePhrase(
                base_phrase.head.global_index,
                [morpheme.global_index for morpheme in base_phrase.morphemes],
                [morpheme.text for morpheme in base_phrase.morphemes],
                is_target=False,
            )
            for base_phrase in base_phrases
        ]

        for idx in range(len(base_phrases)):
            visual_phrase: PhraseAnnotation = visual_phrases[idx]
            mmref_base_phrase: MMRefBasePhrase = mmref_base_phrases[idx]

            # set a parameter: `is_target`
            mmref_base_phrase.is_target = extractor.is_target(visual_phrase)
            # set parameters: `referent_candidates` and `rel2tags`
            if mmref_base_phrase.is_target:
                rel2tags = extractor.extract_rels(
                    visual_phrase, self.candidates, iou_mapper
                )
                assert isinstance(rel2tags, dict)
                mmref_base_phrase.rel2tags = rel2tags

        return mmref_base_phrases
