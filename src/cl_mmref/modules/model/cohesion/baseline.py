import torch
from omegaconf import ListConfig
from torch import nn
from transformers import AutoModel, PreTrainedModel

from cl_mmref.modules.model.dist import calc_4d_dot_product
from cl_mmref.modules.model.heads import Mlp


class BaselineModel(nn.Module):
    """Naive Baseline Model"""

    def __init__(
        self,
        model_name_or_path: str,
        exophora_referents: ListConfig,
        hidden_dropout_prob: float,
        num_tasks: int,
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

        self.mlp = Mlp(
            in_features=self.pretrained_model.config.hidden_size,
            hidden_features=self.pretrained_model.config.hidden_size,
            out_features=self.pretrained_model.config.hidden_size
            * self.num_relation_types,
            drop=hidden_dropout_prob,
        )

        self.analysis_target_classifier = TokenBinaryClassificationHead(
            num_tasks=num_tasks,
            encoder_hidden_size=self.pretrained_model.config.hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
        )

    def forward(
        self,
        input_ids: torch.Tensor,  # (b, seq)
        attention_mask: torch.Tensor,  # (b, seq)
        token_type_ids: torch.Tensor,  # (b, seq)
        **_,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:  # (b, rel, seq, seq), (b, task, seq), (b, seq, rel, hid), (b, seq, rel, hid)
        encoder_last_hidden_state = self.pretrained_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state  # (b, seq, hid)
        batch_size, sequence_len, hidden_size = encoder_last_hidden_state.size()

        hidden_state = self.mlp(encoder_last_hidden_state)  # (b, seq, rel*hid)
        hidden_state = hidden_state.view(
            batch_size, sequence_len, self.num_relation_types, hidden_size
        )  # (b, seq, rel, hid)
        relation_logits = calc_4d_dot_product(
            hidden_state.permute(0, 2, 1, 3), hidden_state.permute(0, 2, 1, 3)
        )  # (b, seq, rel, hid) -> (b, rel, seq, seq)
        source_mask_logits = self.analysis_target_classifier(encoder_last_hidden_state)

        return relation_logits, source_mask_logits


class TokenBinaryClassificationHead(nn.Module):
    def __init__(
        self, num_tasks: int, encoder_hidden_size: int, hidden_dropout_prob: float
    ) -> None:
        super().__init__()
        self.num_tasks = num_tasks
        self.encoder_hidden_size = encoder_hidden_size
        hidden_size = self.encoder_hidden_size
        self.dense = nn.Linear(self.encoder_hidden_size, hidden_size * self.num_tasks)
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
