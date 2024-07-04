from abc import ABC
from typing import Any

import torch
from torchmetrics import Metric


class BaseModuleMetric(Metric, ABC):
    full_state_update = False
    STATE_NAMES: tuple[str, ...]

    def __init__(self) -> None:
        super().__init__()
        # Metric state variables can either be torch.Tensor or an empty list which can be used to store torch.Tensors`.
        # i.e. Expected metric state to be either a Tensor or a list of Tensor
        for state_name in self.STATE_NAMES:
            self.add_state(state_name, default=[], dist_reduce_fx="cat")

    def update(self, kwargs: dict[str, torch.Tensor]) -> None:
        for state_name in self.STATE_NAMES:
            state = getattr(self, state_name)
            value = kwargs[state_name]
            # https://github.com/pytorch/pytorch/issues/90245
            if isinstance(value, torch.BoolTensor):
                value = value.long()
            state.append(value)

    def set_properties(self, kwargs: dict[str, Any]) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
