import h5py
from rhoknp.cohesion import (
    ExophoraReferentType,
)

from mmrr.tools.constants import IOU_THRESHOLD
from mmrr.utils.annotation import Phrase2ObjectRelation, PhraseAnnotation
from mmrr.utils.prediction import ObjectFeature

from .base import BaseExtractor, T, U


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
        base_phrase: T,
        candidates: list[ObjectFeature],
        iou_mapper: dict[str, h5py.Group],
    ) -> dict[str, list[int]]:
        # Initialize with gold phrase to object relations
        assert isinstance(
            base_phrase, PhraseAnnotation
        )  # `base_phrase` means vis_phrase
        all_arguments: dict[str, list[int]] = {
            rel_type: []
            for rel_type in self.get_nonidentical_types(base_phrase.relations)
        }
        for rel_type in all_arguments.keys():
            all_arguments[rel_type] = []
            _rel_types = self.get_rel_types(
                [rel_type], include_nonidentical=self.include_nonidentical
            )
            for relation in base_phrase.relations:
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

    def is_target(self, visual_phrase: T) -> bool:
        assert isinstance(visual_phrase, PhraseAnnotation)
        rel_types = self.get_rel_types(
            self.rels, include_nonidentical=self.include_nonidentical
        )
        for rel in visual_phrase.relations:
            if rel.type in rel_types:
                return True
        return False

    @staticmethod
    def is_candidate(unit: U, predicate: U) -> bool:
        raise NotImplementedError

    @staticmethod
    def get_rel_types(rels: list[str], include_nonidentical: bool = False) -> list[str]:
        if include_nonidentical is True:
            # NOTE: 総称名詞を解析に含める
            nonidentical_rels = [r + "≒" for r in rels]
            return rels + nonidentical_rels
        return rels

    @staticmethod
    def get_nonidentical_types(relations: list[Phrase2ObjectRelation]) -> list[str]:
        all_rels = set(relation.type for relation in relations)
        return [rel for rel in all_rels if "≒" not in rel]
