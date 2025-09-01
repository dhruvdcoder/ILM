from typing import Callable, List, Optional, Tuple, TypedDict
import datasets
import torch
from pcdd.utils.nn import masked_sum
from transformers import AutoTokenizer
from tokenizers import Tokenizer, processors
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers import PreTrainedTokenizerFast
from lightning_utilities.core.rank_zero import rank_zero_warn, rank_zero_info


def preprocess_hf_dataset_for_d3pm_using_pretrained_tokenizer(
    dataset: datasets.Dataset, tokenizer: PreTrainedTokenizerFast
) -> datasets.Dataset:
    """Preprocess a Hugging Face dataset for D3PM using a pretrained tokenizer from transformers."""

    def preprocess(examples):
        # Tokenize the texts
        encodings = tokenizer(examples["text"], add_special_tokens=True)
        return encodings

    return dataset.map(preprocess, batched=True, remove_columns=["text"])


def preprocess_hf_dataset_for_d3pm(
    dataset: datasets.Dataset, tokenizer: Tokenizer
) -> datasets.Dataset:
    """Preprocess a Hugging Face dataset for D3PM using a custom tokenizer trained using the hugging face tokenizer library."""

    def preprocess(examples):
        # Tokenize the texts
        encodings = tokenizer.encode_batch(examples["text"])
        encoding_dict = {
            "input_ids": [encoding.ids for encoding in encodings],
        }
        return encoding_dict

    return dataset.map(preprocess, batched=True, remove_columns=["text"])


# def tokenizer_from_file(
#    tokenizer_path: str,
#    eos_token: str = "<|eos|>",
#    pad_token: str = "<|pad|>",
#    mask_token: str = "<|mask|>",
#    unk_token: str = "<|unk|>",
#    bos_token: str = "<|bos|>",
# ) -> Tokenizer:
#    """Create a tokenizers.Tokenizer from a file."""
#    tokenizer = Tokenizer.from_file(tokenizer_path)
#    tokenizer.eos_token = eos_token
#    tokenizer.pad_token = pad_token
#    tokenizer.mask_token = mask_token
#    tokenizer.unk_token = unk_token
#    tokenizer.bos_token = bos_token
#    tokenizer.add_special_tokens(
#        [eos_token, pad_token, mask_token, unk_token, bos_token]
#    )
#    tokenizer.eos_token_id = tokenizer.token_to_id(eos_token)
#    tokenizer.pad_token_id = tokenizer.token_to_id(pad_token)
#    tokenizer.mask_token_id = tokenizer.token_to_id(mask_token)
#    tokenizer.unk_token_id = tokenizer.token_to_id(unk_token)
#    tokenizer.bos_token_id = tokenizer.token_to_id(bos_token)
#    tokenizer.num_special_tokens = 5
#    # check if all special tokens are at the beginning of the vocab
#    assert all(
#        tokenizer.token_to_id(token) < tokenizer.num_special_tokens
#        for token in [eos_token, pad_token, mask_token, unk_token, bos_token]
#    )
#    return tokenizer


def tokenizer_from_file(
    tokenizer_path: str,
    eos_token: str = "<|eos|>",
    pad_token: str = "<|pad|>",
    mask_token: str = "<|mask|>",
    unk_token: Optional[str] = None,
    bos_token: Optional[str] = None,
    add_eos_post_processor: bool = True,
) -> PreTrainedTokenizerFast:
    """Create a tokenizers.Tokenizer from a file."""
    _tokenizer = Tokenizer.from_file(tokenizer_path)
    if add_eos_post_processor:
        _tokenizer.post_processor = create_eos_post_processor(
            eos_token, _tokenizer.token_to_id(eos_token)
        )
    # ref: https://github.com/huggingface/tokenizers/issues/777
    # TODO: Set block size for the model and the tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=_tokenizer,
        eos_token=eos_token,
        pad_token=pad_token,
        mask_token=mask_token,
        unk_token=unk_token,
        bos_token=bos_token,
    )
    return tokenizer


def create_eos_post_processor(eos_token: str, eos_token_id: int):
    """Create a post processor for the tokenizer to add EOS token to the end of the sequence.

    See https://huggingface.co/docs/tokenizers/v0.13.4.rc2/en/pipeline#postprocessing and
    https://huggingface.co/docs/tokenizers/v0.13.4.rc2/en/api/post-processors#tokenizers.processors.TemplateProcessing
    for more details.
    """
    post_processor = processors.TemplateProcessing(
        single=f"$0 {eos_token}",
        pair=f"$A:0 $B:1 {eos_token}",
        special_tokens=[(eos_token, eos_token_id)],
    )
    return post_processor


class HFBatch(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


def create_collate_fn_for_hf_data_using_pretrained_tokenizer(
    tokenizer: PreTrainedTokenizerFast,
    *args,
    return_tensors="pt",
    padding=True,
    **kwargs,
):
    # make sure that the tokenizer has pad token
    def collate_fn(batch: List[HFBatch]):
        return pad_without_fast_tokenizer_warning(
            tokenizer,
            batch,
            *args,
            return_tensors=return_tensors,
            padding=padding,
            **kwargs,
        )

    return collate_fn


def create_collate_fn_for_hf_data(
    tokenizer: PreTrainedTokenizerFast,
    *args,
    return_tensors="pt",
    padding=True,
    **kwargs,
) -> Callable[[List[HFBatch]], HFBatch]:
    # make sure that the tokenizer has pad token
    def collate_fn(batch: List[HFBatch]) -> HFBatch:
        input_ids, lengths = zip(
            *[(item["input_ids"], len(item["input_ids"])) for item in batch]
        )
        max_length = max(lengths)
        padded_input_ids = torch.tensor(
            [
                item + [tokenizer.pad_token_id] * (max_length - length)
                for item, length in zip(input_ids, lengths)
            ],
            dtype=torch.long,
        )
        return {"input_ids": padded_input_ids}

    return collate_fn


def sample_time_uniform(batch_size: int, T: int):
    """Sample time step uniformly from 1 to T."""
    return torch.randint(1, T, (batch_size,)).long()
    # TODO: Implement sample_time_cosine, sample_time_sigmoid, sample_time_importance


def q_absorb_linear(
    x_0: torch.Tensor, t: torch.Tensor, T: int, mask_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward diffusion process with absorbing noise.

    Args:
        x_0: Input tensor of shape (batch_size, seq_len)
        t: Time step tensor of shape (batch_size,)
        T: Total number of time steps
    Returns:
        Noised tensor of shape (batch_size, seq_len)
    """
    # masked with probability t/T
    # torch.compile friendly
    mask = torch.rand(x_0.shape, device=x_0.device) < t.unsqueeze(-1) / T
    x_t = x_0 * (~mask) + mask * mask_id

    return x_t, mask


def masked_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.BoolTensor
):
    """Compute the cross entropy loss for the masked tokens."""
    labels[~mask] = -100
    loss = torch.nn.functional.cross_entropy(
        logits, labels, reduction="sum", ignore_index=-100
    )
    batch_size = mask.shape[0]
    loss = loss / batch_size
    return loss


def cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.BoolTensor
):
    """Compute the cross entropy loss for the masked tokens."""
    # the mask is ignored here.
    logits = logits.transpose(1, 2)  # shape (bsz, vocab_size, seq_len)
    loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
    batch_size = mask.shape[0]
    loss = loss / batch_size
    return loss
