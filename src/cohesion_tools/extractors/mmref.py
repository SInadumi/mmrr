from rhoknp.cohesion import (
    ExophoraReferentType,
)

from cohesion_tools.extractors.base import BaseExtractor, T
from utils.annotation import PhraseAnnotation
from utils.dataset import ObjectFeature


class MMRefExtractor(BaseExtractor):
    def __init__(
        self,
        cases: list[str],
        exophora_referent_types: list[ExophoraReferentType],
    ) -> None:
        super().__init__(exophora_referent_types)
        self.cases: list[str] = cases

    def extract_rels(
        self,
        predicate: PhraseAnnotation,
        candidates: list[ObjectFeature],
    ) -> dict[str, list[int]]:
        all_arguments: dict[str, list[int]] = {}

        for case in self.cases:
            class_ids = set()
            for relation in predicate.relations:
                if case != relation.type:
                    continue
                class_ids.add(relation.classId)
            all_arguments[case] = [
                idx
                for idx, c in enumerate(candidates)
                if c.class_id.item() in list(class_ids)
            ]

        return all_arguments

    def is_target(self, visual_phrase: dict[str, list]) -> bool:
        return self.is_pas_target(visual_phrase.relations)

    def is_pas_target(self, relations: list[dict]):
        for rel in relations:
            if rel.type in self.cases:
                return True
        return False

    @staticmethod
    def is_candidate(unit: T, predicate: T) -> bool:
        raise NotImplementedError
