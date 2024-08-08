from dataclasses import dataclass

from cohesion_tools.task import Task
from utils.annotation import ImageInfo
from utils.util import CamelCaseDataClassJsonMixin, Rectangle


@dataclass(eq=True)
class BoundingBoxPrediction(CamelCaseDataClassJsonMixin):
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
