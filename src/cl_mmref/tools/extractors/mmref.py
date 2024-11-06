import h5py
from rhoknp.cohesion import (
    ExophoraReferentType,
)

from cl_mmref.tools.constants import IOU_THRESHOLD
from cl_mmref.utils.annotation import PhraseAnnotation
from cl_mmref.utils.prediction import ObjectFeature

from .base import BaseExtractor, T


class MMRefExtractor(BaseExtractor):
    def __init__(
        self,
        rels: list[str],
        exophora_referent_types: list[ExophoraReferentType],
        include_nonidentical: bool = False,
    ) -> None:
        super().__init__(exophora_referent_types)
        self.rels: list[str] = rels
        self.include_nonidentical = include_nonidentical

    def extract_rels(
        self,
        predicate: PhraseAnnotation,
        candidates: list[ObjectFeature],
        iou_mapper: dict[str, h5py.Group],
    ) -> dict[str, list[int]]:
        all_arguments: dict[str, list[int]] = {}
        for rel_type in self.rels:
            all_arguments[rel_type] = []
            _rel_types = self.get_rel_types(
                [rel_type], include_nonidentical=self.include_nonidentical
            )
            for relation in predicate.relations:
                if relation.type not in _rel_types:
                    continue
                if relation.boundingBoxes is None or len(relation.boundingBoxes) == 0:
                    continue
                assert len(candidates) == len(iou_mapper[f"{relation.instanceId}"])
                all_arguments[rel_type] = [
                    idx
                    for idx, iou in enumerate(iou_mapper[f"{relation.instanceId}"])
                    if iou > IOU_THRESHOLD
                ]

        return all_arguments

    def is_target(self, visual_phrase: dict[str, list]) -> bool:
        rel_types = self.get_rel_types(
            self.rels, include_nonidentical=self.include_nonidentical
        )
        for rel in visual_phrase.relations:
            if rel.type in rel_types:
                return True
        return False

    @staticmethod
    def is_candidate(unit: T, predicate: T) -> bool:
        raise NotImplementedError

    @staticmethod
    def get_rel_types(rels: list[str], include_nonidentical: bool = False) -> list[str]:
        if include_nonidentical is True:
            # NOTE: 総称名詞を解析に含める
            nonidentical_rels = [r + "≒" for r in rels]
            return rels + nonidentical_rels
        return rels
