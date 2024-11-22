from dataclasses import dataclass

import torch

from mmrr.utils.util import IGNORE_ID, CamelCaseDataClassJsonMixin, Rectangle


@dataclass
class ObjectFeature:
    feature: torch.Tensor
    image_id: int = IGNORE_ID
    class_id: int = IGNORE_ID
    confidence: float = 0.0
    rect: Rectangle = Rectangle(x1=0, y1=0, x2=0, y2=0)


@dataclass(eq=True)
class BoundingBoxPrediction(CamelCaseDataClassJsonMixin):
    image_id: str
    class_id: int
    rect: Rectangle
    confidence: float


@dataclass(frozen=True, eq=True)
class RelationPrediction(CamelCaseDataClassJsonMixin):
    type: str  # ガ, ヲ, ニ, ノ, =, etc...
    bounding_boxes: list[BoundingBoxPrediction]


@dataclass
class PhrasePrediction(CamelCaseDataClassJsonMixin):
    sid: str
    text: str
    relations: list[RelationPrediction]


@dataclass
class SentencePrediction(CamelCaseDataClassJsonMixin):
    text: str
    sid: str
    phrases: list[PhrasePrediction]


@dataclass
class MMRefPrediction(CamelCaseDataClassJsonMixin):
    doc_id: str
    image_id: str
    phrases: list[PhrasePrediction]
