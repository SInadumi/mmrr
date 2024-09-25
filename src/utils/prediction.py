from dataclasses import dataclass

import torch

from cohesion_tools.task import Task
from utils.annotation import ImageInfo
from utils.util import CamelCaseDataClassJsonMixin, Rectangle

DEFAULT_VIS_EMB_SIZE = 1024


@dataclass(frozen=True)
class ObjectFeature:
    image_id: int = -1
    class_id: torch.Tensor = torch.Tensor([-1.0])
    score: torch.Tensor = torch.Tensor([0.0])
    bbox: torch.Tensor = torch.zeros(4)
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
    bounding_box: list[BoundingBoxPrediction]


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
class UtterancePrediction(CamelCaseDataClassJsonMixin):
    text: str
    sids: list[str]
    phrases: list[PhrasePrediction]


@dataclass
class MMRefPrediction(CamelCaseDataClassJsonMixin):
    scenario_id: str
    images: list[ImageInfo]
    utterances: list[UtterancePrediction]
