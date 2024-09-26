from dataclasses import dataclass

from utils.annotation import Rectangle

RECALL_TOP_KS = (1, 5, 10)
IOU_THRESHOLD = 0.5
CASES: list[str] = (
    "ガ ヲ ニ ト デ カラ ヨリ ヘ マデ ガ２ ヲ２ ニ２ ト２ デ２ カラ２ ヨリ２ ヘ２ マデ２".split()
)
RELATION_TYPES: list[str] = CASES + "ノ ノ？ 修飾 トイウ =".split()


def box_iou(box1: Rectangle, box2: Rectangle) -> float:
    if box1.area == 0 or box2.area == 0:
        return 0
    intersect: int = (box1 & box2).area
    return intersect / (box1.area + box2.area - intersect)


@dataclass
class F1Metric:
    """A data class to calculate and represent F-score"""

    tp_fp: int = 0
    tp_fn: int = 0
    tp: int = 0

    def __add__(self, other: "F1Metric") -> "F1Metric":
        return F1Metric(
            self.tp_fp + other.tp_fp, self.tp_fn + other.tp_fn, self.tp + other.tp
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        return (
            (self.tp_fp == other.tp_fp)
            and (self.tp_fn == other.tp_fn)
            and (self.tp == other.tp)
        )

    @property
    def precision(self) -> float:
        if self.tp_fp == 0:
            return 0.0
        return self.tp / self.tp_fp

    @property
    def recall(self) -> float:
        if self.tp_fn == 0:
            return 0.0
        return self.tp / self.tp_fn

    @property
    def f1(self) -> float:
        if (self.tp_fp + self.tp_fn) == 0:
            return 0.0
        return (2 * self.tp) / (self.tp_fp + self.tp_fn)
