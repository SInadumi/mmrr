from dataclasses import dataclass

import torch

from cl_mmref.tools.task import Task
from cl_mmref.utils.util import CamelCaseDataClassJsonMixin, Rectangle

DEFAULT_VIS_EMB_SIZE = 1024


@dataclass(frozen=True)
class ObjectFeature:
    image_id: int = -1
    class_id: int = -1
    confidence: float = 0.0
    rect: Rectangle = Rectangle(x1=0, y1=0, x2=0, y2=0)
    feature: torch.Tensor = torch.zeros(DEFAULT_VIS_EMB_SIZE)


@dataclass(eq=True)
class BoundingBoxPrediction(CamelCaseDataClassJsonMixin):
    image_id: int
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
    task: Task
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
