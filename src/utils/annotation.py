# https://docs.pydantic.dev/latest/
from pydantic.dataclasses import (
    dataclass,
)

from utils.util import CamelCaseDataClassJsonMixin, Rectangle


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


@dataclass(frozen=True)
class BoundingBox(CamelCaseDataClassJsonMixin):
    imageId: str
    instanceId: str
    rect: Rectangle
    className: str
    classId: int


@dataclass(frozen=True)
class Phrase2ObjectRelation(CamelCaseDataClassJsonMixin):
    type: str  # ガ, ヲ, ニ, ノ, =, etc...
    instanceId: str
    classId: int


@dataclass(frozen=True)
class PhraseAnnotation(CamelCaseDataClassJsonMixin):
    text: str
    relations: list[Phrase2ObjectRelation]


@dataclass(frozen=True)
class ImageAnnotation(CamelCaseDataClassJsonMixin):
    imageId: str
    boundingBoxes: list[BoundingBox]


@dataclass(frozen=True)
class UtteranceAnnotation(CamelCaseDataClassJsonMixin):
    sid: str
    text: str
    phrases: list[PhraseAnnotation]


@dataclass(frozen=True)
class ImageTextAnnotation(CamelCaseDataClassJsonMixin):
    scenarioId: str
    images: list[ImageAnnotation]
    utterances: list[UtteranceAnnotation]
