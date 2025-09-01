from typing import List, Optional, Tuple
import torch
from .configuration_xlnet import XLNetConfig
from .modeling_xlnet import XLNetLMHeadModel, XLNetLMHeadModelOutput


class XLNetTransformer(torch.nn.Module):
    """Wrapper around the huggingface XLNet model."""

    def __init__(
        self,
        num_embeddings: int,  # vocab plus mask and padding other special tokens
        d_model: int,
        num_layers: int,
        nhead: int,
        dim_feedforward: Optional[int] = None,
        padding_idx: int = 0,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        force_flash_attn: bool = False,
    ):
        super().__init__()
        dim_feedforward = dim_feedforward or 4 * d_model
        self.hf_xlnet_config = XLNetConfig(
            vocab_size=num_embeddings,
            d_model=d_model,
            n_layer=num_layers,
            n_head=nhead,
            d_inner=dim_feedforward,
            ff_activation=activation,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
        )
        self._model = XLNetLMHeadModel(self.hf_xlnet_config)

    @property
    def dtype(self) -> torch.dtype:
        return self._model.dtype

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        perm_mask: Optional[torch.Tensor] = None,
        target_mapping: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = self.dtype
        attention_mask = attention_mask.to(dtype=input_ids.dtype)
        if perm_mask is not None:
            perm_mask = perm_mask.to(dtype=dtype)
        if target_mapping is not None:
            target_mapping = target_mapping.to(dtype=dtype)
        hf_output: XLNetLMHeadModelOutput = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            labels=labels,
        )
        return hf_output.loss, hf_output.logits

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
