import torch
from torch import nn

from cl_mmref.modules.model.components.modules import LoRADelta, Mlp
from cl_mmref.modules.model.dist import calc_4d_dot_product


class RelationWiseWordSelectionHead(nn.Module):
    def __init__(
        self,
        hidden_dropout_prob: float,
        hidden_size: int,
        num_relations: int,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=(hidden_size * num_relations) // 2,
            out_features=hidden_size * num_relations,
            drop=hidden_dropout_prob,
        )
        self.num_relation_types = num_relations

    def forward(
        self,
        input_hidden_state: torch.Tensor,  # (b, seq, hid)
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:  # (b, rel, seq, seq)
        batch_size, sequence_len, hidden_size = input_hidden_state.size()
        hidden_state = self.mlp(input_hidden_state)  # (b, seq, rel*hid)
        hidden_state = hidden_state.view(
            batch_size, sequence_len, self.num_relation_types, hidden_size
        )  # (b, seq, rel, hid)
        relation_logits = calc_4d_dot_product(
            hidden_state.permute(0, 2, 1, 3), hidden_state.permute(0, 2, 1, 3)
        )  # (b, seq, rel, hid) -> (b, rel, seq, seq)
        return relation_logits


class LoRARelationWiseWordSelectionHead(nn.Module):
    def __init__(
        self,
        num_relations: int,
        hidden_size: int,
        rank: int = 2,
    ) -> None:
        super().__init__()
        self.delta_parameters = LoRADelta(num_relations, hidden_size, rank)

    def forward(
        self,
        input_hidden_state: torch.Tensor,  # (b, seq, hid)
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:  # (b, rel, seq, seq)
        delta_out = torch.einsum(
            "bsh,hil->bsli", input_hidden_state, self.delta_parameters()
        )  # (b, seq, rel, hid)
        hidden_state = input_hidden_state.unsqueeze(2) + delta_out  # (b, seq, rel, hid)
        relation_logits = calc_4d_dot_product(
            hidden_state.permute(0, 2, 1, 3), hidden_state.permute(0, 2, 1, 3)
        )
        return relation_logits
