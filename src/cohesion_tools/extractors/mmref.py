from rhoknp.cohesion import (
    ExophoraReferentType,
)

from cohesion_tools.extractors.base import BaseExtractor, T
from utils.annotation import PhraseAnnotation
from utils.dataset import ObjectFeature


class MMRefExtractor(BaseExtractor):
    def __init__(
        self,
        rels: list[str],
        exophora_referent_types: list[ExophoraReferentType],
    ) -> None:
        super().__init__(exophora_referent_types)
        self.rels: list[str] = rels

    def extract_rels(
        self,
        predicate: PhraseAnnotation,
        candidates: list[ObjectFeature],
    ) -> dict[str, list[int]]:
        all_arguments: dict[str, list[int]] = {}

        # TODO: Make a distinction between ≒ or not
        for rel_type in self.rels:
            class_ids = set()
            for relation in predicate.relations:
                if rel_type != relation.type:
                    continue
                class_ids.add(relation.classId)
            all_arguments[rel_type] = [
                idx
                for idx, c in enumerate(candidates)
                if c.class_id.item() in list(class_ids)
            ]

        return all_arguments

    def is_target(self, visual_phrase: dict[str, list]) -> bool:
        for rel in visual_phrase.relations:
            # TODO: Make a distinction between ≒ or not
            if rel.type in self.rels:
                return True
        return False

    @staticmethod
    def is_candidate(unit: T, predicate: T) -> bool:
        raise NotImplementedError
