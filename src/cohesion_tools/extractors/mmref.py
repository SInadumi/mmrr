from collections import defaultdict
from typing import List

from rhoknp.cohesion import (
    ExophoraReferentType,
)

from cohesion_tools.extractors.base import BaseExtractor, T


class MMRefExtractor(BaseExtractor):
    def __init__(
        self,
        cases: List[str],
        exophora_referent_types: List[ExophoraReferentType],
    ) -> None:
        super().__init__(exophora_referent_types)
        self.cases: List[str] = cases

    def extract_rels(
        self, predicate: dict[str, list], candidates: dict, is_neg: bool = False
    ) -> dict[str, list[str]]:
        all_arguments: dict[str, list[str]] = defaultdict(list)
        for case in self.cases:
            cat_ids = set()

            for relation in predicate.relations:
                if case != relation.type:
                    continue
                cat_ids.add(relation.classId)

            # return difference set between candidate and gold data
            if is_neg:
                cat_ids = set(candidates.keys()) - cat_ids

            all_arguments[case] = [candidates[cid] for cid in list(cat_ids)]
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
