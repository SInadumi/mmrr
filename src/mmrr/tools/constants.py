from rhoknp.cohesion import ExophoraReferent, ExophoraReferentType

RECALL_TOP_KS = (-1, 1, 5, 10)
IOU_THRESHOLD = 0.5
CASES: list[str] = (
    "ガ ヲ ニ ト デ カラ ヨリ ヘ マデ ガ２ ヲ２ ニ２ ト２ デ２ カラ２ ヨリ２ ヘ２ マデ２".split()
)
RELATION_TYPES: list[str] = CASES + "ノ ノ？ 修飾 トイウ =".split()
EXOPHORA_REFERENTS: list[ExophoraReferentType] = [
    ExophoraReferent(s) for s in ["著者", "読者", "不特定:人", "不特定:物"]
]
EXOPHORA_REFERENT_TYPES: list[ExophoraReferentType] = [
    er.type for er in EXOPHORA_REFERENTS
]
