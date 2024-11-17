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
        self.delta_parameters = LoRADelta(num_relations, hidden_size, hidden_size, rank)

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


class RelationWiseObjectSelectionHeads(nn.Module):
    def __init__(
        self,
        hidden_dropout_prob: float,
        source_hidden_size: int,
        target_hidden_size: int,
        num_relations: int,
    ) -> None:
        super().__init__()
        self.output_size = source_hidden_size
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.l_source = Mlp(
            in_features=source_hidden_size,
            hidden_features=(source_hidden_size * num_relations) // 2,
            out_features=self.output_size * num_relations,
            drop=hidden_dropout_prob,
        )
        self.l_target = Mlp(
            in_features=target_hidden_size,
            hidden_features=(target_hidden_size * num_relations) // 2,
            out_features=self.output_size * num_relations,
            drop=hidden_dropout_prob,
        )
        self.num_relation_types = num_relations

    def forward(
        self,
        source_hidden_state: torch.Tensor,  # (b, seq, hid_source)
        target_hidden_state: torch.Tensor,  # (b, seq, hid_target)
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:  # (b, rel, seq, seq)
        batch_size, sequence_len, hidden_size = source_hidden_state.size()
        assert hidden_size == self.output_size

        h_source = self.l_source(source_hidden_state)
        h_source = h_source.view(
            batch_size, sequence_len, self.num_relation_types, self.output_size
        )  # (b, seq, hid_source) -> (b, seq, rel*hid_source) -> (b, seq, rel, hid_source)
        h_target = self.l_target(target_hidden_state)
        h_target = h_target.view(
            batch_size, sequence_len, self.num_relation_types, self.output_size
        )  # (b, seq, hid_target) -> (b, seq, rel*hid_target) -> (b, seq, rel, hid_source)

        relation_logits = calc_4d_dot_product(
            h_source.permute(0, 2, 1, 3), h_target.permute(0, 2, 1, 3)
        )  # (b, seq, rel, hid) -> (b, rel, seq, seq)
        return relation_logits


class LoRARelationWiseObjectSelectionHeads(nn.Module):
    def __init__(
        self,
        num_relations: int,
        source_hidden_size: int,
        target_hidden_size: int,
        rank: int = 2,
    ) -> None:
        super().__init__()
        self.output_size = source_hidden_size
        self.l_source = nn.Linear(source_hidden_size, self.output_size)
        self.l_target = nn.Linear(target_hidden_size, self.output_size)
        self.delta_source = LoRADelta(
            num_labels=num_relations,
            input_hidden_size=source_hidden_size,
            output_hidden_size=self.output_size,
            rank=rank,
        )
        self.delta_target = LoRADelta(
            num_labels=num_relations,
            input_hidden_size=target_hidden_size,
            output_hidden_size=self.output_size,
            rank=rank,
        )

    def forward(
        self,
        source_hidden_state: torch.Tensor,  # (b, seq, hid_source)
        target_hidden_state: torch.Tensor,  # (b, seq, hid_target)
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:  # (b, rel, seq, seq)
        h_source = self.l_source(source_hidden_state)  # -> (b, seq, hid_source)
        delta_source_out = torch.einsum(
            "bsh,hil->bsli", source_hidden_state, self.delta_source()
        )  # (b, seq, hid_source) -> (b, seq, rel, hid_source)
        source_out = (
            h_source.unsqueeze(2) + delta_source_out
        )  # (b, seq, rel, hid_source)

        h_target = self.l_target(target_hidden_state)
        delta_target_out = torch.einsum(
            "bsh,hil->bsli", target_hidden_state, self.delta_target()
        )  # (b, seq, hid_target) -> (b, seq, rel, hid_source)
        target_out = (
            h_target.unsqueeze(2) + delta_target_out
        )  # (b, seq, rel, hid_source)

        relation_logits = calc_4d_dot_product(
            source_out.permute(0, 2, 1, 3), target_out.permute(0, 2, 1, 3)
        )
        return relation_logits
