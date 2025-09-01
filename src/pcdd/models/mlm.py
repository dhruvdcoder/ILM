# v2
from typing import Optional, Tuple

import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Integer
from torch import Tensor as TT

from pcdd.modules.rotary_transformer import (
    RotaryTransformerFinalLayer,
    RotaryTransformerFinalLayerForClassification,
    RotaryTransformerLayer,
    RotaryTransformerLayerList,
    RotaryEmbedding,
)
from pcdd.modules.gpt2_transformer import GPT, GPTConfig

from .model import Model
from pcdd.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

########################################################
# region: Rotary Transformer


class BaseRotaryTransformerMLMModel(torch.nn.Module, Model):
    "Rotary embedding based transformer decoder."

    def __init__(
        self,
        num_embeddings: int,  # vocab plus mask and padding other special tokens
        d_model: int,
        num_layers: int,
        nhead: int,
        padding_idx: int = 0,
        mask_idx: int = 1,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        rotary_emb_dim: int = 64,
        max_length: int = 1024,
        force_flash_attn: bool = False,
        final_layer_without_normalization: bool = True,
        input_constraint: bool = False,
        keep_type_embeddings: bool = True,  # URGENT: Setting this to True so the running preemptable jobs don't fail
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.mask_idx = mask_idx
        self.embed_tokens = nn.Embedding(
            num_embeddings, d_model, padding_idx=padding_idx
        )
        self.dim_feedforward = dim_feedforward or 4 * d_model
        encoder_layer = RotaryTransformerLayer(
            d_model,
            nhead,
            self.dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            force_flash_attn=force_flash_attn,
        )
        self.max_length = max_length
        self.encoder = RotaryTransformerLayerList.from_layer(
            encoder_layer,
            num_layers,
            RotaryEmbedding(
                rotary_emb_dim, head_first=True, cache_size=max_length
            ),
        )
        self.output_layer = RotaryTransformerFinalLayer(
            d_model,
            num_embeddings,
            layer_norm_eps,
            use_final_layer_norm=not final_layer_without_normalization,
        )
        self.num_embeddings = num_embeddings
        self.input_constraint = input_constraint
        if input_constraint:
            self.type_embeddings = nn.Embedding(
                2, d_model, padding_idx=0
            )  # padding_idx will be initialized to 0 and get no gradient
        elif keep_type_embeddings:
            # to load models with type embeddings from checkpoints with optimizer states
            # we could rely on the else block below when loading only model weights
            # but optimizer won't load correctly
            self.type_embeddings = nn.Embedding(
                2, d_model, padding_idx=0
            )  # padding_idx will be initialized to 0 and get no gradient

        else:
            self.type_embeddings = None

    def forward(
        self,
        x_t: Integer[TT, " *batch seq_len"],
        attention_mask: Optional[Bool[TT, " *batch seq_len"]] = None,
        positions: Optional[Integer[TT, " *batch seq_len"]] = None,
        token_type_ids: Optional[Integer[TT, " *batch seq_len"]] = None,
    ) -> Tuple[
        Float[TT, " *batch seq_len vocab_size"],
        Float[TT, " *batch max_length"],
    ]:
        """
        Args:
            x_t: The input tokens of shape (*batch, seq_len)
            t: The timesteps of shape (*batch)
            attention_mask: The attention mask of shape (*batch, seq_len), which is True for non-padding tokens.
            positions: The positions of the tokens of shape (*batch, seq_len)
        """
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)

        x = self.embed_tokens(x_t)  # shape (batch_size, seq_len, d_model)
        if token_type_ids is not None:
            if self.input_constraint:
                x = x + self.type_embeddings(token_type_ids.to(torch.long))
            else:
                raise ValueError(
                    "token_type_ids are provided but input_constraint is False"
                )

        for block in self.encoder:
            x = block(x, attention_mask, positions=positions)

        vocab_logits = self.output_layer(
            x,
        )  # shape (batch_size, seq_len, vocab_size)
        return x, vocab_logits

    def get_named_params_for_weight_decay(self):
        # all parameters except biases and layer-norm parameters
        for name, param in self.named_parameters():
            if "bias" in name or "norm" in name:
                continue
            yield (name, param)

    def get_named_params_for_no_weight_decay(self):
        # biases and layer-norm parameters
        for name, param in self.named_parameters():
            if "bias" in name or "norm" in name:
                yield (name, param)
