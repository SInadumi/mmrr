import torch
from omegaconf import ListConfig
from torch import nn
from transformers import AutoModel, PreTrainedModel


class BaselineModel(nn.Module):
    """Naive Baseline Model"""

    def __init__(
        self,
        model_name_or_path: str,
        exophora_referents: ListConfig,
        hidden_dropout_prob: float,
        num_tasks: int,
        vis_emb_size: int,
        num_relations: int,
        **_,
    ) -> None:
        super().__init__()
        self.pretrained_model: PreTrainedModel = AutoModel.from_pretrained(
            model_name_or_path
        )
        self.pretrained_model.resize_token_embeddings(
            self.pretrained_model.config.vocab_size
            + len(exophora_referents)
            + 2,  # +2: [NULL] and [NA]
        )

        self.dropout = nn.Dropout(hidden_dropout_prob)

        self.num_relation_types = num_relations
        hidden_size = self.pretrained_model.config.hidden_size

        self.l_source = nn.Linear(
            hidden_size,
            hidden_size * self.num_relation_types,
        )
        self.l_target = nn.Linear(
            vis_emb_size,
            hidden_size * self.num_relation_types,
        )
        self.out = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,  # (b, seq)
        attention_mask: torch.Tensor,  # (b, seq)
        token_type_ids: torch.Tensor,  # (b, seq)
        vis_embeds: torch.Tensor,  # (b, seq)
        **_,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:  # (b, rel, seq, seq), (b, task, seq), (b, seq, rel, hid), (b, seq, rel, hid)
        encoder_last_hidden_state = self.pretrained_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state  # (b, seq, hid)
        batch_size, txsequence_len, hidden_size = encoder_last_hidden_state.size()
        _, vis_sequence_len, _ = vis_embeds.size()

        h_src = self.l_source(
            self.dropout(encoder_last_hidden_state)
        )  # (b, seq, rel*hid)
        h_tgt = self.l_target(self.dropout(vis_embeds))  # (b, seq, rel*hid)
        h_src = h_src.view(
            batch_size, txsequence_len, self.num_relation_types, hidden_size
        )  # (b, seq, rel, hid)
        h_tgt = h_tgt.view(
            batch_size, vis_sequence_len, self.num_relation_types, hidden_size
        )  # (b, seq, rel, hid)
        h = torch.tanh(
            self.dropout(h_src.unsqueeze(2) + h_tgt.unsqueeze(1))
        )  # (b, seq, seq, rel, hid)
        # -> (b, seq, seq, rel, 1) -> (b, seq, seq, rel) -> (b, rel, seq, seq)
        relation_logits = self.out(h).squeeze(-1).permute(0, 3, 1, 2).contiguous()

        return relation_logits, h_src, h_tgt
