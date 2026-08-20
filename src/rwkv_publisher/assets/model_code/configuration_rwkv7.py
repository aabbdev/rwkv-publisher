# Copyright 2026 The RWKV team and The HuggingFace Inc. team. All rights reserved.
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
"""RWKV-7 (Goose) model configuration."""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(
    checkpoint="RWKV/RWKV7-1.5B-20260805",
    custom_intro="""
    Configuration for [`Rwkv7Model`], an all-recurrent (attention-free) RWKV-7 "Goose"
    model. Instantiating with the defaults yields the ~0.1B RWKV-7 configuration.

    Parameter names follow the upstream RWKV reference implementation
    (`BlinkDL/RWKV-LM`) rather than a renamed variant.
    """,
)
@strict
class Rwkv7Config(PreTrainedConfig):
    r"""
    vocab_size (`int`, *optional*, defaults to 65536):
        Vocabulary size (RWKV "world" tokenizer).
    hidden_size (`int`, *optional*, defaults to 768):
        Model width `C`.
    num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of blocks.
    head_dim (`int`, *optional*, defaults to 64):
        Width of one WKV head. `hidden_size` must be divisible by it.
    num_heads (`int`, *optional*, defaults to 12):
        Number of WKV heads; must equal `hidden_size // head_dim`.
    decay_low_rank_dim (`int`, *optional*):
        Rank of the decay (`w`) LoRA. When omitted, follows the RWKV-LM recipe:
        `max(32, round(2.5 * sqrt(hidden_size) / 32) * 32)`.
    a_low_rank_dim (`int`, *optional*):
        Rank of the in-context-learning-rate (`a`) LoRA. Uses the same default
        formula as `decay_low_rank_dim`.
    v_low_rank_dim (`int`, *optional*):
        Rank of the value-residual (`v`) LoRA. Defaults to
        `max(32, round(1.7 * sqrt(hidden_size) / 32) * 32)`. Unused on layer 0,
        which *produces* `v_first` instead of mixing towards it.
    gate_low_rank_dim (`int`, *optional*):
        Rank of the output-gate (`g`) LoRA. Defaults to
        `max(32, round(5 * sqrt(hidden_size) / 32) * 32)`.
    intermediate_size (`int`, *optional*):
        Channel-mix inner width. Defaults to `4 * hidden_size`.
    norm_eps (`float`, *optional*, defaults to 1e-05):
        Epsilon of every LayerNorm/GroupNorm in the model.
    norm_bias (`bool`, *optional*, defaults to `True`):
        Whether the norms carry a bias.
    tie_word_embeddings (`bool`, *optional*, defaults to `False`):
        Whether to tie the input embedding and the LM head.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether to return the recurrent state.
    wkv_state_dtype (`str`, *optional*, defaults to `"float32"`):
        Precision the recurrent WKV state is carried and accumulated in,
        independently of the activation dtype. The recurrence is unrolled over
        the whole sequence, so a narrow state drifts; `"float32"` with fp16
        activations is the combination the reference implementation uses.
        `"float16"`/`"bfloat16"` trade that for a smaller state.
    wkv_implementation (`str`, *optional*, defaults to `"eager"`):
        Which WKV recurrence to use, by name, from
        `models.rwkv7.modeling_rwkv7.RWKV7_WKV_FUNCTIONS`. `"eager"` is the
        exact portable PyTorch path and preserves the reference token order for
        both prefill and decode. `"chunked"` is an explicit faster
        chunk-parallel rearrangement whose floating-point order differs. Register
        an entry in that mapping to plug in a fused or varlen kernel without
        forking the model.
    bos_token_id (`int`, *optional*, defaults to 0):
        Beginning-of-sequence id. The RWKV world tokenizer has no dedicated BOS
        token and the reference implementation prepends nothing, so this exists to
        satisfy `GenerationMixin` rather than to be emitted.
    eos_token_id (`int`, *optional*, defaults to 0):
        End-of-sequence id, id 0 in the RWKV world vocabulary.
    pad_token_id (`int`, *optional*, defaults to 0):
        Padding id, the same id 0. Set deliberately rather than left `None`:
        `generate` needs one to pad a batch, and without it a batched call either
        raised or fell back to the eos id with a warning on every step.

    ```python
    >>> from transformers import Rwkv7Config, Rwkv7Model

    >>> configuration = Rwkv7Config()
    >>> model = Rwkv7Model(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "rwkv7"
    keys_to_ignore_at_inference = ["state"]

    vocab_size: int = 65536
    hidden_size: int = 768
    num_hidden_layers: int = 12
    head_dim: int = 64
    num_heads: int = 12
    decay_low_rank_dim: int | None = None
    a_low_rank_dim: int | None = None
    v_low_rank_dim: int | None = None
    gate_low_rank_dim: int | None = None
    # `None` rather than a number: a literal default is correct for the default
    # `hidden_size` and silently wrong for every other one, so a config built as
    # `Rwkv7Config(hidden_size=4096, num_heads=64)` would come back with a channel-mix
    # four times narrower than the architecture it names. `__post_init__` resolves it,
    # and the resolved value is written to `config.json` either way.
    intermediate_size: int | None = None
    norm_eps: float = 1e-5
    norm_bias: bool = True
    tie_word_embeddings: bool = False
    use_cache: bool = True
    wkv_state_dtype: str = "float32"
    wkv_implementation: str = "eager"
    bos_token_id: int | None = 0
    eos_token_id: int | None = 0
    pad_token_id: int | None = 0

    def __post_init__(self, **kwargs):
        rank_unit = self.hidden_size**0.5 / 32
        if self.decay_low_rank_dim is None:
            self.decay_low_rank_dim = max(32, round(2.5 * rank_unit) * 32)
        if self.a_low_rank_dim is None:
            self.a_low_rank_dim = max(32, round(2.5 * rank_unit) * 32)
        if self.v_low_rank_dim is None:
            self.v_low_rank_dim = max(32, round(1.7 * rank_unit) * 32)
        if self.gate_low_rank_dim is None:
            self.gate_low_rank_dim = max(32, round(5 * rank_unit) * 32)
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size
        if self.wkv_state_dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError(f"wkv_state_dtype must be float32/float16/bfloat16, got {self.wkv_state_dtype}")
        if self.hidden_size % self.head_dim != 0:
            raise ValueError(f"hidden_size {self.hidden_size} must be divisible by head_dim {self.head_dim}")
        if self.num_heads != self.hidden_size // self.head_dim:
            raise ValueError(
                f"num_heads must be hidden_size // head_dim = {self.hidden_size // self.head_dim}, "
                f"got {self.num_heads}"
            )
        super().__post_init__(**kwargs)


__all__ = ["Rwkv7Config"]
