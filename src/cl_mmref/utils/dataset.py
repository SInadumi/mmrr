from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class CohesionBasePhrase:
    """A wrapper class of BasePhrase for cohesion analysis"""

    head_morpheme_global_index: int  # a phrase head index
    morpheme_global_indices: list[int]  # indices of phrase span
    morphemes: list[str]  # phrase span
    is_target: bool  # a flag of phrase span an analysis target
    referent_candidates: list["CohesionBasePhrase"]
    rel2tags: Optional[dict[str, list[str]]] = None


@dataclass(frozen=True)
class CohesionInputFeatures:
    """A dataclass which represents a raw model input.

    The attributes of this class correspond to arguments of forward method of each model.
    """

    example_id: int
    input_ids: list[int]
    attention_mask: list[bool]
    token_type_ids: list[int]
    source_mask: list[
        bool
    ]  # loss を計算する対象の基本句かどうか（文書分割によって文脈としてのみ使用される場合は False）
    target_mask: list[
        list[list[bool]]
    ]  # source と関係を持つ候補かどうか（後ろと共参照はしないなど）
    source_label: list[list[int]]  # 解析対象基本句かどうか
    target_label: list[list[list[float]]]  # source と関係を持つかどうか


@dataclass
class MMRefBasePhrase:
    """A wrapper class of BasePhrase for multi-modal reference resolution"""

    head_morpheme_global_index: int  # a phrase head index
    morpheme_global_indices: list[int]  # indices of phrase span
    morphemes: list[str]  # phrase span
    is_target: bool  # a flag of phrase span an analysis target
    rel2tags: Optional[dict[str, list[int]]] = None


@dataclass(frozen=True)
class MMRefInputFeatures:
    """A dataclass which represents a language encoder and interaction layer input"""

    example_id: int
    input_ids: list[int]
    attention_mask: list[bool]
    token_type_ids: list[int]
    source_mask: list[bool]  # loss を計算する対象の基本句かどうか
    source_label: list[list[int]]  # 解析対象基本句かどうか
    vis_embeds: torch.Tensor
    vis_attention_mask: list[bool]
    target_mask: list[list[list[bool]]]
    target_label: list[list[list[float]]]
