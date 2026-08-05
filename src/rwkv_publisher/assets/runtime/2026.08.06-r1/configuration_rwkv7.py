from __future__ import annotations

from typing import Any, Mapping

import torch
from transformers import PretrainedConfig


class RWKV7Config(PretrainedConfig):
    """Configuration for RWKV-7 causal language models."""

    model_type = "rwkv7"
    keys_to_ignore_at_inference = ("past_key_values",)

    def __init__(
        self,
        vocab_size: int = 65536,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        head_size: int = 64,
        intermediate_size: int | None = None,
        decay_lora_rank: int = 64,
        a_lora_rank: int = 64,
        gate_lora_rank: int = 128,
        value_lora_rank: int = 32,
        layer_norm_epsilon: float = 1e-5,
        use_cache: bool = True,
        kernel_backend: str = "auto",
        recurrent_state_dtype: str = "float32",
        rwkv_prefill_chunk_size: int = 256,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = False,
        bos_token_id: int | None = None,
        eos_token_id: int | None = 0,
        pad_token_id: int | None = 0,
        **kwargs: Any,
    ) -> None:
        if hidden_size % head_size:
            raise ValueError("hidden_size must be divisible by head_size")
        if kernel_backend not in {"auto", "torch", "tilelang"}:
            raise ValueError(f"Unsupported kernel backend: {kernel_backend}")
        if recurrent_state_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError(
                "recurrent_state_dtype must be float32, float16, or bfloat16"
            )
        if rwkv_prefill_chunk_size < 0:
            raise ValueError("rwkv_prefill_chunk_size must be non-negative")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.head_size = head_size
        self.num_attention_heads = hidden_size // head_size
        self.intermediate_size = intermediate_size or hidden_size * 4
        self.decay_lora_rank = decay_lora_rank
        self.a_lora_rank = a_lora_rank
        self.gate_lora_rank = gate_lora_rank
        self.value_lora_rank = value_lora_rank
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_cache = use_cache
        self.kernel_backend = kernel_backend
        self.recurrent_state_dtype = recurrent_state_dtype
        self.rwkv_prefill_chunk_size = rwkv_prefill_chunk_size
        self.initializer_range = initializer_range
        self.is_decoder = True
        self.is_encoder_decoder = False
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )

    @classmethod
    def from_rwkv_weights(
        cls, weights: Mapping[str, torch.Tensor], **kwargs: Any
    ) -> "RWKV7Config":
        embedding = weights["emb.weight"]
        hidden_size = embedding.shape[1]
        layer_ids: set[int] = set()
        for name in weights:
            if not name.startswith("blocks."):
                continue
            try:
                layer_ids.add(int(name.split(".")[1]))
            except (IndexError, ValueError):
                continue
        if not layer_ids:
            raise ValueError("Checkpoint contains no RWKV blocks")
        head_size = weights["blocks.0.att.r_k"].shape[1]
        inferred: dict[str, Any] = {
            "vocab_size": embedding.shape[0],
            "hidden_size": hidden_size,
            "num_hidden_layers": max(layer_ids) + 1,
            "head_size": head_size,
            "intermediate_size": weights["blocks.0.ffn.key.weight"].shape[0],
            "decay_lora_rank": weights["blocks.0.att.w1"].shape[1],
            "a_lora_rank": weights["blocks.0.att.a1"].shape[1],
            "gate_lora_rank": weights["blocks.0.att.g1"].shape[1],
            "value_lora_rank": weights["blocks.1.att.v1"].shape[1]
            if max(layer_ids) >= 1
            else max(1, hidden_size // 24),
        }
        inferred.update(kwargs)
        return cls(**inferred)
