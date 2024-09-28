from typing import Optional

# https://docs.pydantic.dev/latest/
from pydantic.dataclasses import (
    dataclass,
)

from .util import CamelCaseDataClassJsonMixin, Rectangle

IGNORE_CLASS_ID = -1


@dataclass
class ImageInfo(CamelCaseDataClassJsonMixin):
    id: str
    path: str
    time: int


@dataclass
class UtteranceInfo(CamelCaseDataClassJsonMixin):
    text: str
    sids: list[str]
    start: int
    end: int
    duration: int
    speaker: str
    image_ids: list[str]

    @property
    def image_indices_span(self) -> tuple[int, int]:
        # zero origin
        return (int(self.image_ids[0]) - 1, int(self.image_ids[-1]) - 1)


@dataclass
class DatasetInfo(CamelCaseDataClassJsonMixin):
    scenario_id: str
    utterances: list[UtteranceInfo]
    images: list[ImageInfo]


@dataclass
class BoundingBox(CamelCaseDataClassJsonMixin):
    imageId: str
    instanceId: str
    rect: Rectangle
    className: str
    classId: int = IGNORE_CLASS_ID


@dataclass
class Phrase2ObjectRelation(CamelCaseDataClassJsonMixin):
    type: str  # ガ, ヲ, ニ, ノ, =, etc...
    instanceId: str
    classId: int = IGNORE_CLASS_ID
    boundingBoxes: Optional[list[BoundingBox]] = None  # 発話区間に含まれる bboxes


@dataclass
class PhraseAnnotation(CamelCaseDataClassJsonMixin):
    text: str
    relations: list[Phrase2ObjectRelation]


@dataclass
class ImageAnnotation(CamelCaseDataClassJsonMixin):
    imageId: str
    boundingBoxes: list[BoundingBox]


@dataclass
class SentenceAnnotation(CamelCaseDataClassJsonMixin):
    text: str
    phrases: list[PhraseAnnotation]
    sid: Optional[str] = None


@dataclass
class ImageTextAnnotation(CamelCaseDataClassJsonMixin):
    scenarioId: str
    images: list[ImageAnnotation]
    utterances: list[SentenceAnnotation]
