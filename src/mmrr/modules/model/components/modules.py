import math
from typing import Optional

import torch
from torch import nn

from .activations import ACT2FN


class LoRADelta(nn.Module):
    def __init__(
        self,
        num_labels: int,
        input_hidden_size: int,
        output_hidden_size: int,
        rank: int,
    ) -> None:
        super().__init__()
        self.dense_a = nn.Parameter(torch.Tensor(input_hidden_size, rank, num_labels))
        self.dense_b = nn.Parameter(torch.Tensor(rank, output_hidden_size, num_labels))
        nn.init.kaiming_uniform_(self.dense_a, a=math.sqrt(5))
        nn.init.zeros_(self.dense_b)

    def forward(self) -> torch.Tensor:
        return torch.einsum(
            "hrl,ril->hil", self.dense_a, self.dense_b
        )  # (hid, hid, label)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer="gelu",
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = ACT2FN[act_layer]
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# coding=utf-8
# Copyright 2024 IDEA Research and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Grounding DINO model."""

class GroundingDinoMultiheadAttention(nn.Module):
    """Equivalent implementation of nn.MultiheadAttention with `batch_first=True`. HACK: modified from original class"""

    def __init__(
        self,
        hidden_size: int,
        attention_dropout: float,
        num_attention_heads: int,
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(attention_dropout)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> tuple[torch.Tensor, ...]:
        query_layer = self.transpose_for_scores(self.query(queries))
        key_layer = self.transpose_for_scores(self.key(keys))
        value_layer = self.transpose_for_scores(self.value(values))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in GroundingDinoModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        context_layer = self.out_proj(context_layer)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return outputs


class GroundingDinoDecoderLayer(nn.Module):
    """HACK: modified from original class"""
    def __init__(
        self,
        d_model: int,
        decoder_attention_heads: int = 8,
        dropout: float = 0.1,
        activation_function: str = "relu",
        activation_dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        decoder_ffn_dim: int = 2048,
    ):
        super().__init__()

        # self-attention
        self.self_attn = GroundingDinoMultiheadAttention(
            hidden_size=d_model,
            attention_dropout=dropout,
            num_attention_heads=decoder_attention_heads,
        )

        self.dropout = dropout
        self.activation_fn = ACT2FN[activation_function]
        self.activation_dropout = activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(d_model, layer_norm_eps)
        # cross-attention text
        self.encoder_attn_text = GroundingDinoMultiheadAttention(
            hidden_size=d_model,
            attention_dropout=dropout,
            num_attention_heads=decoder_attention_heads,
        )
        self.encoder_attn_text_layer_norm = nn.LayerNorm(d_model, layer_norm_eps)
        self.encoder_attn_layer_norm = nn.LayerNorm(d_model, layer_norm_eps)
        # feedforward neural networks
        self.fc1 = nn.Linear(d_model, decoder_ffn_dim)
        self.fc2 = nn.Linear(decoder_ffn_dim, d_model)
        self.final_layer_norm = nn.LayerNorm(d_model, layer_norm_eps)

    def forward(
        self,
        vision_encoder_hidden_states: torch.Tensor,
        text_encoder_hidden_states: torch.Tensor,
        vision_encoder_attention_mask: Optional[torch.Tensor] = None,
        text_encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    )-> tuple[torch.Tensor, ...]:
        residual = vision_encoder_hidden_states
        # Self Attention Vision
        queries = keys = vision_encoder_hidden_states
        hidden_states, self_attn_weights = self.self_attn(
            queries=queries,
            keys=keys,
            values=vision_encoder_hidden_states,
            attention_mask=vision_encoder_attention_mask,
            output_attentions=True,
        )

        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        second_residual = hidden_states

        # Cross-Attention Text
        queries = hidden_states
        hidden_states, text_cross_attn_weights = self.encoder_attn_text(
            queries=queries,
            keys=text_encoder_hidden_states,
            values=text_encoder_hidden_states,
            attention_mask=text_encoder_attention_mask,
            output_attentions=True,
        )

        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = second_residual + hidden_states
        hidden_states = self.encoder_attn_text_layer_norm(hidden_states)

        third_residual = hidden_states

        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = third_residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,self_attn_weights, text_cross_attn_weights) if output_attentions else (hidden_states,)

        return outputs
