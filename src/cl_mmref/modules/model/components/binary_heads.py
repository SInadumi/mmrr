import math

import torch
from torch import nn

from cl_mmref.modules.model.components.modules import LoRADelta


class TokenBinaryClassificationHead(nn.Module):
    def __init__(
        self, num_tasks: int, hidden_size: int, hidden_dropout_prob: float
    ) -> None:
        super().__init__()
        self.num_tasks = num_tasks
        self.dense = nn.Linear(hidden_size, hidden_size * self.num_tasks)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(
        self,
        hidden_state: torch.Tensor,  # (b, seq, hid)
    ) -> torch.Tensor:  # (b, task, seq)
        batch_size, sequence_len, hidden_size = hidden_state.size()
        h = self.dense(self.dropout(hidden_state))  # (b, seq, task*hid)
        h = h.view(
            batch_size, sequence_len, self.num_tasks, hidden_size
        )  # (b, seq, task, hid)
        # -> (b, seq, task, 1) -> (b, seq, task) -> (b, task, seq)
        return (
            self.classifier(torch.tanh(self.dropout(h)))
            .squeeze(-1)
            .permute(0, 2, 1)
            .contiguous()
        )


class LoRARelationWiseTokenBinaryClassificationHead(nn.Module):
    def __init__(
        self,
        num_tasks: int,
        hidden_size: int,
        hidden_dropout_prob: float,
        rank: int,
    ) -> None:
        super().__init__()
        self.num_tasks = num_tasks
        self.dense = nn.Linear(hidden_size, hidden_size * self.num_tasks)
        self.delta = LoRADelta(num_tasks, hidden_size, hidden_size, rank)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Parameter(torch.Tensor(hidden_size, num_tasks))
        nn.init.kaiming_uniform_(self.classifier, a=math.sqrt(5))

    def forward(
        self,
        hidden_state: torch.Tensor,  # (b, seq, hid)
    ) -> torch.Tensor:  # (b, task, seq)
        batch_size, sequence_len, hidden_size = hidden_state.size()
        h = self.dense(self.dropout(hidden_state))  # (b, seq, task*hid)
        h = h.view(
            batch_size, sequence_len, self.num_tasks, hidden_size
        )  # (b, seq, task, hid)
        delta_out = torch.einsum(
            "bsh,hil->bsli", hidden_state, self.delta()
        )  # (b, seq, task, hid)
        # -> (b, seq, task, 1) -> (b, seq, task) -> (b, task, seq)
        hidden = torch.tanh(self.dropout(h + delta_out))  # (b, seq, task, hid)
        # (b, seq, task) -> (b, task, seq)
        return (
            torch.einsum("bsth,ht->bst", hidden, self.classifier)
            .permute(0, 2, 1)
            .contiguous()
        )
