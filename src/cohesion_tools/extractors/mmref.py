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
        for rel_type in self.rels:
            _rel_types = self.get_rel_types([rel_type], include_nonidentical=False)
            class_ids = set()
            for relation in predicate.relations:
                if relation.type not in _rel_types:
                    continue
                class_ids.add(relation.classId)
            all_arguments[rel_type] = [
                idx
                for idx, c in enumerate(candidates)
                if c.class_id.item() in list(class_ids)
            ]
        return all_arguments

    def is_target(self, visual_phrase: dict[str, list]) -> bool:
        rel_types = self.get_rel_types(self.rels, include_nonidentical=False)
        for rel in visual_phrase.relations:
            if rel.type in rel_types:
                return True
        return False

    @staticmethod
    def is_candidate(unit: T, predicate: T) -> bool:
        raise NotImplementedError

    @staticmethod
    def get_rel_types(rels: list[str], include_nonidentical: bool = False):
        if include_nonidentical is True:
            nonidentical_rels = [r + "≒" for r in rels]
            return rels + nonidentical_rels
        return rels
