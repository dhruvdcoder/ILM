from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float, Integer
from torch import Tensor as TT
from torch import nn

from pcdd import flags
from pcdd.modules.ddit_simple import AdaLNModulations, LayerNormAndScale
from pcdd.utils.nn import masked_mean

from .d3pm import AbsorbingDDiTD3PMModel, Model


class DDitVarLenModel(AbsorbingDDiTD3PMModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with torch.no_grad():
            self.output_layer.linear.weight.normal_(0.0, 1.0)

    def forward(
        self,
        x_t: Integer[TT, " *batch seq_len"],
        t: Integer[TT, " *batch"],
        attention_mask: Optional[Bool[TT, " *batch seq_len"]] = None,
        positions: Optional[Integer[TT, " *batch seq_len"]] = None,
    ) -> Tuple[
        Float[TT, " *batch seq_len vocab_size"],
        Float[TT, " *batch seq_len d_model"],
    ]:
        """
        Args:
            x_t: The input tokens of shape (*batch, seq_len)
            t: The timesteps of shape (*batch)
            attention_mask: The attention mask of shape (*batch, seq_len), which is True for non-padding tokens.
        """
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)

        x = self.embed_tokens(x_t)  # shape (batch_size, seq_len, d_model)
        # TODO: Why do we need to use silu here?
        c = F.silu(self.sigma_map(t))

        if not flags.DEBUG_PENULTIMATE_LAYER:
            for block in self.encoder:
                x = block(x, c, attention_mask, positions=positions)
            y = x
        else:
            for i, block in enumerate(self.encoder):
                if i == len(self.encoder) - 1:
                    y = block(x, c, attention_mask, positions=positions)
                else:
                    x = block(x, c, attention_mask, positions=positions)

        logits = self.output_layer(
            x, c
        )  # shape (batch_size, seq_len, vocab_size)
        return logits, y


class LearnedSoftPlus(torch.nn.Module):
    def __init__(self, init_beta=1.0, threshold=20):
        super().__init__()
        # keep beta > 0
        self.log_beta_bias = torch.nn.Parameter(
            torch.tensor(float(init_beta)).log()
        )
        self.threshold = threshold

    def forward(self, x: Float[TT, " *batch"]) -> Float[TT, " *batch"]:
        beta = self.log_beta_bias.exp()
        beta_x = beta * x
        return torch.where(
            beta_x < self.threshold, torch.log1p(beta_x.exp()) / beta, x
        )


class LogitsLengthModel(Model):
    def __init__(self, num_embeddings: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.shift_bias = nn.Parameter(torch.zeros(1, 1, self.num_embeddings))
        self.scale_bias = nn.Parameter(torch.ones(1, 1, self.num_embeddings))
        with torch.no_grad():
            self.scale_bias.fill_(1e-1)
        self.activation = LearnedSoftPlus()

    def forward(
        self, x: Float[TT, " *batch seq_len vocab_size"]
    ) -> Float[TT, " *batch"]:
        a = self.activation(x * self.scale_bias + self.shift_bias).sum(
            dim=(1, 2)
        )  # shape (batch,)
        return a


class EmbeddingLengthModel(Model):
    def __init__(
        self,
        max_output_length: int,
        input_dim: int,
        d_model: int = 128,
        num_layers: int = 4,
        nhead: int = 8,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        max_length: int = 1024,
    ):
        super().__init__()
        self.dim_feedforward = dim_feedforward or 4 * d_model
        self.project_in = nn.Linear(input_dim, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=True,
        )
        self.max_length = max_length
        self.max_output_length = max_output_length
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers,
        )
        self.output_layer = nn.Linear(d_model, max_output_length, bias=False)

    def forward(
        self,
        h: Float[TT, " *batch seq_len hidden_dim"],
        attention_mask: Optional[Bool[TT, " *batch seq_len"]] = None,
    ) -> Float[TT, " *batch"]:
        """
        Args:
            h: The input hidden states of shape (*batch, seq_len, hidden_dim)
            attention_mask: The attention mask of shape (*batch, seq_len), which is True for non-padding tokens.
        """
        if attention_mask is not None:
            attention_mask = attention_mask.to(
                torch.bool
            )  # (batch_size, seq_len)

        x = self.project_in(h)  # shape (batch_size, seq_len, d_model)
        x = self.encoder(
            x,
            src_key_padding_mask=~attention_mask,  # TransformerEncoder expects 1 for pad tokens
        )  # shape (batch_size, seq_len, d_model)
        # mean embedding
        x = masked_mean(
            x, attention_mask.unsqueeze(-1), dim=1  # type: ignore
        )  # shape (batch_size, d_model)

        logits = self.output_layer(x)  # shape (batch_size, max_output_length)
        return logits

    def get_mean_delta_l(
        self, length_logits: Float[TT, " *batch max_output_length"]
    ) -> Float[TT, " *batch"]:
        delta_l = torch.arange(
            0, self.max_output_length, device=length_logits.device
        )
        p = torch.softmax(length_logits, dim=-1)
        return (p * delta_l).sum(dim=-1)  # shape (*batch)


class EmbeddingLengthModel2(Model):
    def __init__(
        self,
        max_output_length: int,
        input_dim: int,
        d_model: int = 128,
        num_layers: int = 4,
        nhead: int = 8,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        max_length: int = 1024,
    ):
        super().__init__()
        self.dim_feedforward = dim_feedforward or 4 * d_model
        # self.project_in = nn.Linear(input_dim, d_model)
        self.embed_length = nn.Embedding(max_output_length, d_model)
        self.project_in = nn.Sequential(
            nn.Linear(input_dim, 2 * input_dim),
            nn.SiLU(),
            nn.Linear(2 * input_dim, d_model),
        )
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=True,
        )
        self.max_length = max_length
        self.max_output_length = max_output_length
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers,
        )
        self.output_layer = nn.Linear(d_model, max_output_length, bias=False)
        with torch.no_grad():
            self.output_layer.weight = self.embed_length.weight

    def forward(
        self,
        h: Float[TT, " *batch seq_len hidden_dim"],
        attention_mask: Optional[Bool[TT, " *batch seq_len"]] = None,
    ) -> Float[TT, " *batch"]:
        """
        Args:
            h: The input hidden states of shape (*batch, seq_len, hidden_dim)
            attention_mask: The attention mask of shape (*batch, seq_len), which is True for non-padding tokens.
        """
        if attention_mask is not None:
            attention_mask = attention_mask.to(
                torch.bool
            )  # (batch_size, seq_len)
        assert attention_mask is not None
        current_length = attention_mask.sum(dim=-1)
        x = self.project_in(h)  # shape (batch_size, seq_len, d_model)
        x = self.encoder(
            x,
            src_key_padding_mask=~attention_mask,  # TransformerEncoder expects 1 for pad tokens
        )  # shape (batch_size, seq_len, d_model)
        # mean embedding
        x = masked_mean(
            x, attention_mask.unsqueeze(-1), dim=1  # type: ignore
        )  # shape (batch_size, d_model)
        current_l_emb = self.embed_length(
            current_length
        )  # shape (batch_size, d_model)
        x = x + current_l_emb
        absolute_length_logits = self.output_layer(
            x
        )  # shape (batch_size, max_output_length)
        # reassign the logits to delta_l = absolute_length - current_length
        deltas = (
            torch.arange(
                0,
                # self.max_output_length + 1, # allow 0 as well as max_output_length
                self.max_output_length,
                device=absolute_length_logits.device,
            )
            .unsqueeze(0)
            .expand_as(absolute_length_logits)
        )  # shape (batch_size, max_output_length)
        # compute the absolute indices and place them in deltas' positions
        absolute_indices = deltas + current_length.unsqueeze(
            -1
        )  # shape (batch_size, max_output_length)
        # Note: the values in absolute_indices will exceed the max_output_length
        # create mask to mark valid deltas
        # mask = absolute_indices <= self.max_output_length # allow 0 as well as max_output_length
        mask = absolute_indices < self.max_output_length
        delta_l_logits = torch.full_like(absolute_length_logits, -10e7)
        delta_l_logits[mask] = absolute_length_logits.gather(
            1,
            # absolute_indices.clamp(max=self.max_output_length) # allow 0 as well as max_output_length
            absolute_indices.clamp(max=self.max_output_length - 1),
        )[mask]
        logits = self.output_layer(x)  # shape (batch_size, max_output_length)
        return logits

    def get_mean_delta_l(
        self, length_logits: Float[TT, " *batch max_output_length"]
    ) -> Float[TT, " *batch"]:
        delta_l = torch.arange(
            0, self.max_output_length, device=length_logits.device
        )
        p = torch.softmax(length_logits, dim=-1)
        return (p * delta_l).sum(dim=-1)  # shape (*batch)


class SimpleEmbeddingLengthModel(Model):
    def __init__(
        self,
        max_output_length: int,
        input_dim: int,
        d_model: int = 128,
        d_cond: int = 128,
    ):
        super().__init__()
        self.max_output_length = max_output_length
        # self.embed_length = nn.Embedding(
        #    max_output_length + 1, d_model
        # )  # allow 0 as well as max_output_length
        self.embed_length = nn.Embedding(max_output_length, d_model)
        # two layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 2 * input_dim),
            nn.SiLU(),
            nn.Linear(2 * input_dim, d_model),
        )
        self.layer_norm = LayerNormAndScale(d_model)
        self.adaLN_modulation = AdaLNModulations(
            d_cond, d_model, num_modulation_parameters=2
        )
        self.output_layer = nn.Linear(d_model, max_output_length, bias=False)
        # share the weights of the output layer with the embedding layer
        with torch.no_grad():
            self.output_layer.weight = self.embed_length.weight
            # initialize MLP to zero
            self.mlp[2].weight.data.zero_()
            self.mlp[2].bias.data.zero_()

    def forward(
        self,
        h: Float[TT, " *batch seq_len hidden_dim"],
        attention_mask: Optional[Bool[TT, " *batch seq_len"]] = None,
        current_length: Optional[Integer[TT, " *batch"]] = None,
    ) -> Float[TT, " *batch"]:
        """
        Args:
            h: The input hidden states of shape (*batch, seq_len, hidden_dim)
            attention_mask: The attention mask of shape (*batch, seq_len), which is True for non-padding tokens.
            current_length: The current length of the sequence of shape (*batch)
        """
        if attention_mask is not None:
            attention_mask = attention_mask.to(
                torch.bool
            )  # (batch_size, seq_len)
        else:
            attention_mask = torch.ones(
                (h.shape[0], h.shape[1]), dtype=torch.bool
            )

        if current_length is None:
            if attention_mask is None:
                # CLEANUP: Remove this
                _current_length = h.shape[1]
                current_length = torch.full_like(
                    h[:, 0, 0], _current_length, dtype=torch.long
                )
            else:
                current_length = attention_mask.sum(dim=-1)
        out = self.embed_length(current_length)  # shape (batch_size, d_model)
        x = self.mlp(h)  # shape (batch_size, seq_len, d_model)
        # out = out + x.sum(dim=1)  # shape (batch_size, d_model)
        out = out + (x * attention_mask.unsqueeze(-1)).sum(
            dim=1
        )  # shape (batch_size, d_model)
        absolute_length_logits = self.output_layer(
            out
        )  # shape (batch_size, max_output_length + 1)
        # reassign the logits to delta_l = absolute_length - current_length
        deltas = (
            torch.arange(
                0,
                # self.max_output_length + 1, # allow 0 as well as max_output_length
                self.max_output_length,
                device=absolute_length_logits.device,
            )
            .unsqueeze(0)
            .expand_as(absolute_length_logits)
        )  # shape (batch_size, max_output_length)
        # compute the absolute indices and place them in deltas' positions
        absolute_indices = deltas + current_length.unsqueeze(
            -1
        )  # shape (batch_size, max_output_length)
        # Note: the values in absolute_indices will exceed the max_output_length
        # create mask to mark valid deltas
        # mask = absolute_indices <= self.max_output_length # allow 0 as well as max_output_length
        mask = absolute_indices < self.max_output_length
        delta_l_logits = torch.full_like(absolute_length_logits, -10e7)
        delta_l_logits[mask] = absolute_length_logits.gather(
            1,
            # absolute_indices.clamp(max=self.max_output_length) # allow 0 as well as max_output_length
            absolute_indices.clamp(max=self.max_output_length - 1),
        )[mask]
        return delta_l_logits

    def get_mean_delta_l(
        self, length_logits: Float[TT, " *batch max_output_length"]
    ) -> Float[TT, " *batch"]:
        delta_l = torch.arange(
            0, self.max_output_length, device=length_logits.device
        )
        p = torch.softmax(length_logits, dim=-1)
        return (p * delta_l).sum(dim=-1)  # shape (*batch)
