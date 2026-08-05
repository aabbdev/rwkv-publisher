from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_rwkv7 import RWKV7Config
from .custom_ops import tilelang_state_finalize_op
from .kernel_dispatch import resolve_backend  # type: ignore[reportMissingImports]
from .kernel_tilelang_state import tilelang_state_update  # type: ignore[reportMissingImports]
from .state import RWKV7LayerState, RWKV7State  # type: ignore[reportMissingImports]

TORCH_STACK = torch.stack  # type: ignore[attr-defined]
TORCH_WHERE = torch.where  # type: ignore[attr-defined]
FLOAT32 = torch.float32  # type: ignore[attr-defined]
FLOAT16 = torch.float16  # type: ignore[attr-defined]
BFLOAT16 = torch.bfloat16  # type: ignore[attr-defined]
BOOL = torch.bool  # type: ignore[attr-defined]
TORCH_EMPTY = torch.empty  # type: ignore[attr-defined]
TORCH_CAT = torch.__dict__["cat"]
TORCH_ZEROS_LIKE = torch.__dict__["zeros_like"]
IS_GRAD_ENABLED = torch.is_grad_enabled  # type: ignore[attr-defined]
TORCH_COMPILE = torch.compile  # type: ignore[attr-defined]
IS_COMPILING = torch.compiler.is_compiling



def _cache_tensor_values(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        tensors: list[torch.Tensor] = []
        for item in value:
            tensors.extend(_cache_tensor_values(item))
        return tensors
    if value is not None and value.__class__.__name__ == "TileLangDecodeWorkspace":
        tensors = []
        for item in vars(value).values():
            tensors.extend(_cache_tensor_values(item))
        return tensors
    return []


def _unique_storage_bytes(values: list[Any]) -> int:
    storages: dict[tuple[str, int], int] = {}
    for value in values:
        for tensor in _cache_tensor_values(value):
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr())
            storages[key] = max(storages.get(key, 0), storage.nbytes())
    return sum(storages.values())

def _storage_keys(values: list[Any]) -> set[tuple[str, int]]:
    keys = set()
    for value in values:
        for tensor in _cache_tensor_values(value):
            storage = tensor.untyped_storage()
            keys.add((str(tensor.device), storage.data_ptr()))
    return keys


def _unique_storage_bytes_excluding(
    values: list[Any], excluded: set[tuple[str, int]]
) -> int:
    storages: dict[tuple[str, int], int] = {}
    for value in values:
        for tensor in _cache_tensor_values(value):
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr())
            if key not in excluded:
                storages[key] = max(storages.get(key, 0), storage.nbytes())
    return sum(storages.values())


def _replace_parameter_storage(
    parameter: nn.Parameter, tensor: torch.Tensor
) -> None:
    with torch.no_grad():
        parameter.set_(tensor)


def _restore_parameter_storage(parameter: nn.Parameter) -> None:
    _replace_parameter_storage(
        parameter, parameter.detach().clone(memory_format=torch.contiguous_format)
    )


def _reset_elapsed_tokens(
    elapsed_tokens: torch.Tensor | None,
    reset: torch.Tensor | None,
) -> None:
    """Reset per-request token counts before processing a reset position."""
    if elapsed_tokens is None or reset is None:
        return
    elapsed_tokens.masked_fill_(reset.reshape_as(elapsed_tokens), 0)


def _advance_elapsed_tokens(
    elapsed_tokens: torch.Tensor | None,
    active: torch.Tensor | None,
) -> None:
    """Advance each request once for an active token."""
    if elapsed_tokens is None:
        return
    if active is None:
        elapsed_tokens.add_(1)
    else:
        elapsed_tokens.add_(
            active.reshape_as(elapsed_tokens).to(dtype=elapsed_tokens.dtype)
        )


def _sequence_previous(
    x: torch.Tensor,
    initial: torch.Tensor,
    active: torch.Tensor | None,
    reset: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return previous active value for every sequence position."""
    sequence_length = x.shape[1]
    if active is None and reset is None:
        previous = TORCH_CAT((initial.unsqueeze(1), x[:, :-1]), dim=1)
        return previous, x[:, -1]

    previous_value = initial
    previous_steps: list[torch.Tensor] = []
    for token_index in range(sequence_length):
        if reset is not None:
            reset_token = reset[:, token_index]
            previous_value = TORCH_WHERE(
                reset_token, TORCH_ZEROS_LIKE(previous_value), previous_value
            )
        previous_steps.append(previous_value)
        current = x[:, token_index]
        if active is None:
            previous_value = current
        else:
            previous_value = TORCH_WHERE(
                active[:, token_index], current, previous_value
            )
    return TORCH_STACK(previous_steps, dim=1), previous_value


def _sequence_matmul(
    x: torch.Tensor, weight: torch.Tensor, tokenwise: bool
) -> torch.Tensor:
    if not tokenwise:
        return x @ weight
    return TORCH_STACK(
        [x[:, token_index] @ weight for token_index in range(x.shape[1])],
        dim=1,
    )


def _sequence_linear(
    module: nn.Linear, x: torch.Tensor, tokenwise: bool
) -> torch.Tensor:
    if not tokenwise:
        return module(x)
    return TORCH_STACK(
        [module(x[:, token_index]) for token_index in range(x.shape[1])],
        dim=1,
    )


def _reset_layer_state(
    state: RWKV7LayerState, reset: torch.Tensor | None
) -> RWKV7LayerState:
    if reset is None:
        return state
    matrix_reset = reset.unsqueeze(-1).unsqueeze(-1)
    return RWKV7LayerState(
        TORCH_WHERE(reset, TORCH_ZEROS_LIKE(state.channel), state.channel),
        TORCH_WHERE(reset, TORCH_ZEROS_LIKE(state.time_shift), state.time_shift),
        TORCH_WHERE(
            matrix_reset, TORCH_ZEROS_LIKE(state.time_matrix), state.time_matrix
        ),
    )

class RWKV7Attention(nn.Module):
    def __init__(self, config: RWKV7Config, layer_id: int) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.layer_id = layer_id
        self.num_heads = config.num_attention_heads
        self.head_size = config.head_size
        self.kernel_backend = config.kernel_backend
        self.hidden_size = hidden
        self.intermediate_size = config.intermediate_size

        self.x_r = nn.Parameter(TORCH_EMPTY(hidden))
        self.x_w = nn.Parameter(TORCH_EMPTY(hidden))
        self.x_k = nn.Parameter(TORCH_EMPTY(hidden))
        self.x_v = nn.Parameter(TORCH_EMPTY(hidden))
        self.x_a = nn.Parameter(TORCH_EMPTY(hidden))
        self.x_g = nn.Parameter(TORCH_EMPTY(hidden))
        self.w0 = nn.Parameter(TORCH_EMPTY(hidden))
        self.w1 = nn.Parameter(TORCH_EMPTY(hidden, config.decay_lora_rank))
        self.w2 = nn.Parameter(TORCH_EMPTY(config.decay_lora_rank, hidden))
        self.a0 = nn.Parameter(TORCH_EMPTY(hidden))
        self.a1 = nn.Parameter(TORCH_EMPTY(hidden, config.a_lora_rank))
        self.a2 = nn.Parameter(TORCH_EMPTY(config.a_lora_rank, hidden))
        self.g1 = nn.Parameter(TORCH_EMPTY(hidden, config.gate_lora_rank))
        self.g2 = nn.Parameter(TORCH_EMPTY(config.gate_lora_rank, hidden))
        self.k_k = nn.Parameter(TORCH_EMPTY(hidden))
        self.k_a = nn.Parameter(TORCH_EMPTY(hidden))
        self.r_k = nn.Parameter(TORCH_EMPTY(self.num_heads, self.head_size))

        self.v0 = nn.Parameter(TORCH_EMPTY(hidden))
        self.v1 = nn.Parameter(TORCH_EMPTY(hidden, config.value_lora_rank))
        self.v2 = nn.Parameter(TORCH_EMPTY(config.value_lora_rank, hidden))
        if layer_id == 0:
            self.v0.requires_grad_(False)
            self.v1.requires_grad_(False)
            self.v2.requires_grad_(False)

        self.receptance = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.output = nn.Linear(hidden, hidden, bias=False)
        self.ln_x = nn.GroupNorm(
            self.num_heads,
            hidden,
            eps=config.head_size * config.layer_norm_epsilon,
            affine=True,
        )
        self._x_mix_cache: torch.Tensor | None = None
        self._x_mix_cache_versions: tuple[int, ...] | None = None
        self.cuda_graph_time_mix = False
        self._time_mix_graph: Any | None = None
        self._time_mix_graph_key: tuple[Any, ...] | None = None
        self._decode_backend: Any | None = None
        self._decode_workspace: Any | None = None
        self._decode_weight: torch.Tensor | None = None
        self._decode_weight_key: tuple[Any, ...] | None = None
        self._decode_rankout_weights: tuple[torch.Tensor, ...] | None = None
        self._rkv_bmm_weight: torch.Tensor | None = None
        self._rkv_bmm_weight_key: tuple[Any, ...] | None = None

    def _x_mix_weights(self, x: torch.Tensor) -> torch.Tensor:
        parameters = (self.x_r, self.x_w, self.x_k, self.x_v, self.x_a, self.x_g)
        if self.training or IS_GRAD_ENABLED():
            return TORCH_STACK(parameters)
        if IS_COMPILING() and self._x_mix_cache is not None:
            return self._x_mix_cache
        versions = tuple(getattr(parameter, "_version", 0) for parameter in parameters)
        cache = self._x_mix_cache
        if (
            cache is None
            or cache.device != x.device
            or cache.dtype != x.dtype
            or self._x_mix_cache_versions != versions
        ):
            cache = TORCH_STACK(parameters).detach()
            self._x_mix_cache = cache
            self._x_mix_cache_versions = versions
        return cache

    def _packed_rkv_bmm_weight(
        self, reference: torch.Tensor
    ) -> torch.Tensor | None:
        if (
            self.kernel_backend != "tilelang"
            or self.training
            or IS_GRAD_ENABLED()
            or reference.shape != (1, self.hidden_size)
            or reference.dtype != FLOAT16
            or reference.device.type != "cuda"
            or torch.cuda.get_device_capability(reference.device) != (12, 0)
        ):
            return None
        parameters = (
            self.receptance.weight, self.key.weight, self.value.weight
        )
        cache_key = (
            reference.device,
            reference.dtype,
            *(getattr(parameter, "_version", 0) for parameter in parameters),
        )
        if (
            self._rkv_bmm_weight is None
            or self._rkv_bmm_weight_key != cache_key
        ):
            self._rkv_bmm_weight = TORCH_STACK(
                tuple(parameter.t() for parameter in parameters)
            ).contiguous()
            self._rkv_bmm_weight_key = cache_key
        return self._rkv_bmm_weight

    def _tilelang_backend_cache(self, reference: torch.Tensor) -> tuple[Any, Any]:
        if (
            self.kernel_backend != "tilelang"
            or self.training
            or IS_GRAD_ENABLED()
            or reference.shape != (1, self.hidden_size)
            or reference.dtype not in {torch.float16, torch.bfloat16}
            or reference.device.type != "cuda"
            or torch.cuda.get_device_capability(reference.device) != (12, 0)
        ):
            raise RuntimeError("TileLang B1T1 backend is unavailable")
        from rwkv7_pytorch.tilelang_decode import (
            TileLangDecodeBackend,
            TileLangDecodeSpec,
        )

        backend = self._decode_backend
        if (
            backend is None
            or self._decode_workspace is None
            or backend.device != reference.device
            or backend.dtype != reference.dtype
        ):
            spec = TileLangDecodeSpec(
                channels=self.hidden_size,
                ffn_rows=self.intermediate_size,
                num_heads=self.num_heads,
                head_size=self.head_size,
                ranks=(
                    self.w1.size(1),
                    self.a1.size(1),
                    self.g1.size(1),
                    self.v1.size(1),
                ),
            )
            backend = TileLangDecodeBackend(
                spec, reference.device, reference.dtype
            )
            self._decode_backend = backend
            self._decode_workspace = backend.create_workspace()
            self._decode_weight = None
            self._decode_rankout_weights = None
            self._decode_weight_key = None
        return backend, self._decode_workspace

    def _tilelang_layernorm_mix6(
        self,
        residual: torch.Tensor,
        previous: torch.Tensor,
        norm: nn.LayerNorm,
        active: torch.Tensor | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]] | None:
        if (
            self.kernel_backend != "tilelang"
            or self.training
            or IS_GRAD_ENABLED()
            or residual.shape != (1, self.hidden_size)
            or residual.dtype not in {torch.float16, torch.bfloat16}
            or residual.device.type != "cuda"
            or torch.cuda.get_device_capability(residual.device) != (12, 0)
            or active is not None
            or norm.weight is None
            or norm.bias is None
        ):
            return None
        backend, workspace = self._tilelang_backend_cache(residual)
        normalized, mixed = backend.tmix_layernorm_mix6(
            residual.view(-1),
            previous.view(-1),
            norm.weight,
            norm.bias,
            self._x_mix_weights(residual),
            float(norm.eps),
            workspace,
        )
        return (
            normalized.view(1, self.hidden_size),
            tuple(
                value.view(1, self.hidden_size)
                for value in mixed.unbind(dim=0)
            ),
        )

    def _clear_inference_cache(self) -> None:
        packed_weight = self._decode_weight
        if packed_weight is not None:
            packed_keys = _storage_keys([packed_weight])
            for parameter in (
                self.receptance.weight,
                self.key.weight,
                self.value.weight,
                self.w1,
                self.a1,
                self.g1,
                self.v1,
            ):
                if _storage_keys([parameter]) & packed_keys:
                    _restore_parameter_storage(parameter)
        rankout_weights = self._decode_rankout_weights
        if rankout_weights is not None:
            rankout_keys = _storage_keys([rankout_weights])
            for parameter in (self.w2, self.a2, self.g2, self.v2):
                if _storage_keys([parameter]) & rankout_keys:
                    _restore_parameter_storage(parameter)
        self._decode_backend = None
        self._decode_workspace = None
        self._decode_weight = None
        self._decode_rankout_weights = None
        self._decode_weight_key = None

    def _prepare_tilelang_cache(
        self, reference: torch.Tensor
    ) -> tuple[Any, Any, torch.Tensor, tuple[torch.Tensor, ...], bool]:
        from rwkv7_pytorch.tilelang_decode import (
            TileLangDecodeBackend,
            TileLangDecodeSpec,
        )

        parameters = (
            self.receptance.weight,
            self.key.weight,
            self.value.weight,
            self.w1,
            self.a1,
            self.g1,
            self.v1,
            self.w2,
            self.a2,
            self.g2,
            self.v2,
        )
        weight_key = (
            reference.device,
            reference.dtype,
            *(getattr(parameter, "_version", 0) for parameter in parameters),
        )
        created = (
            self._decode_backend is None
            or self._decode_workspace is None
            or self._decode_weight is None
            or self._decode_rankout_weights is None
            or self._decode_weight_key != weight_key
        )
        if created:
            self._clear_inference_cache()
            spec = TileLangDecodeSpec(
                channels=self.hidden_size,
                ffn_rows=self.intermediate_size,
                num_heads=self.num_heads,
                head_size=self.head_size,
                ranks=(
                    self.w1.size(1),
                    self.a1.size(1),
                    self.g1.size(1),
                    self.v1.size(1),
                ),
            )
            backend = TileLangDecodeBackend(
                spec, reference.device, reference.dtype
            )
            dense_parameters = (
                self.receptance.weight, self.key.weight, self.value.weight
            )
            lowrank_parameters = (self.w1, self.a1, self.g1, self.v1)
            packed_weight = backend.pack_rkv_weights(
                dense_parameters,
                tuple(
                    parameter.t().contiguous()
                    for parameter in lowrank_parameters
                ),
            )
            rankout_parameters = (self.w2, self.a2, self.g2, self.v2)
            rankout_weights = tuple(
                parameter.t().contiguous()
                for parameter in rankout_parameters
            )
            start = 0
            for parameter in dense_parameters:
                rows = parameter.size(0)
                _replace_parameter_storage(
                    parameter, packed_weight.narrow(0, start, rows)
                )
                start += rows
            for parameter, rank in zip(
                lowrank_parameters, spec.ranks, strict=True
            ):
                _replace_parameter_storage(
                    parameter, packed_weight.narrow(0, start, rank).t()
                )
                start += rank
            for parameter, packed in zip(
                rankout_parameters, rankout_weights, strict=True
            ):
                _replace_parameter_storage(parameter, packed.t())
            self._decode_backend = backend
            self._decode_workspace = backend.create_workspace()
            self._decode_weight = packed_weight
            self._decode_rankout_weights = rankout_weights
            self._decode_weight_key = (
                reference.device,
                reference.dtype,
                *(
                    getattr(parameter, "_version", 0)
                    for parameter in parameters
                ),
            )
        backend = self._decode_backend
        workspace = self._decode_workspace
        packed_weight = self._decode_weight
        rankout_weights = self._decode_rankout_weights
        if (
            backend is None
            or workspace is None
            or packed_weight is None
            or rankout_weights is None
        ):
            raise RuntimeError("TileLang decode cache initialization failed")
        return backend, workspace, packed_weight, rankout_weights, created

    def _tilelang_projections(
        self,
        xr: torch.Tensor,
        xw: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        xa: torch.Tensor,
        xg: torch.Tensor,
        first_value: torch.Tensor,
    ) -> tuple[torch.Tensor, ...] | None:
        if (
            self.kernel_backend != "tilelang"
            or self.training
            or IS_GRAD_ENABLED()
            or xr.shape != (1, self.hidden_size)
            or xr.dtype not in {torch.float16, torch.bfloat16}
            or xr.device.type != "cuda"
            or torch.cuda.get_device_capability(xr.device) != (12, 0)
        ):
            return None
        backend, workspace, packed_weight, rankout_weights, _ = (
            self._prepare_tilelang_cache(xr)
        )
        backend.rkv(
            tuple(
                value.view(-1) for value in (xr, xk, xv, xw, xa, xg)
            ),
            packed_weight,
            workspace.rkv_output,
        )
        (
            receptance,
            key,
            value_base,
            decay_rank,
            a_rank,
            g_rank,
            value_rank,
        ) = backend.rkv_views(workspace.rkv_output)
        next_first_value = (
            value_base.view(1, -1)
            if self.layer_id == 0
            else first_value
        )
        decay, gate_a, gate_g, value = backend.rankout_reduced(
            (decay_rank, a_rank, g_rank, value_rank),
            rankout_weights,
            (self.a0, self.v0),
            value_base,
            next_first_value.view(-1),
            workspace,
            use_value_mix=self.layer_id != 0,
        )
        return (
            decay.view(1, -1),
            receptance.view(1, -1),
            key.view(1, -1),
            value.view(1, -1),
            gate_a.view(1, -1),
            gate_g.view(1, -1),
            next_first_value,
        )



    def set_cuda_graph_time_mix(self, enabled: bool) -> None:
        self.cuda_graph_time_mix = enabled
        if not enabled:
            self._time_mix_graph = None
            self._time_mix_graph_key = None


    def _cuda_graph_time_mix(
        self,
        x: torch.Tensor,
        previous: torch.Tensor,
        first_value: torch.Tensor,
    ) -> tuple[torch.Tensor, ...] | None:
        if (
            not self.cuda_graph_time_mix
            or self.training
            or IS_GRAD_ENABLED()
            or x.device.type != "cuda"
            or x.shape[0] != 1
        ):
            return None
        versions = tuple(
            getattr(parameter, "_version", 0) for parameter in self.parameters()
        )
        key = (str(x.device), x.dtype, tuple(x.shape), versions)
        if self._time_mix_graph is None or self._time_mix_graph_key != key:

            def graph_function(
                graph_x: torch.Tensor,
                graph_previous: torch.Tensor,
                graph_first_value: torch.Tensor,
            ) -> tuple[torch.Tensor, ...]:
                delta = graph_previous - graph_x
                x_mix = TORCH_STACK(
                    (self.x_r, self.x_w, self.x_k, self.x_v, self.x_a, self.x_g)
                )
                xr, xw, xk, xv, xa, xg = (
                    graph_x.unsqueeze(1) + delta.unsqueeze(1) * x_mix
                ).unbind(dim=1)
                decay = self.w0 + (xw @ self.w1).tanh() @ self.w2
                decay = (-0.606531 * decay.sigmoid()).exp()
                receptance = self.receptance(xr)
                key_tensor = self.key(xk)
                value = self.value(xv)
                if self.layer_id == 0:
                    graph_first_value = value
                else:
                    value = value + (graph_first_value - value) * (
                        self.v0 + (xv @ self.v1) @ self.v2
                    ).sigmoid()
                gate_a = (self.a0 + (xa @ self.a1) @ self.a2).sigmoid()
                gate_g = (xg @ self.g1).sigmoid() @ self.g2
                return (
                    decay,
                    receptance,
                    key_tensor,
                    value,
                    gate_a,
                    gate_g,
                    graph_first_value,
                )

            self._time_mix_graph = TORCH_COMPILE(
                graph_function,
                backend="cudagraphs",
                fullgraph=True,
                dynamic=False,
            )
            self._time_mix_graph_key = key
        return self._time_mix_graph(x, previous, first_value)

    def forward(
        self,
        x: torch.Tensor,
        state: RWKV7LayerState,
        first_value: torch.Tensor,
        active: torch.Tensor | None,
        elapsed_tokens: torch.Tensor | None = None,
        mixed_inputs: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        heads = self.num_heads
        head_size = self.head_size
        decay_delta: torch.Tensor | None = None

        graph_outputs = (
            None
            if mixed_inputs is not None
            else self._cuda_graph_time_mix(x, state.time_shift, first_value)
        )
        if graph_outputs is None:
            if mixed_inputs is None:
                delta = state.time_shift - x
                x_mix = self._x_mix_weights(x)
                xr, xw, xk, xv, xa, xg = (
                    x.unsqueeze(1) + delta.unsqueeze(1) * x_mix
                ).unbind(dim=1)
            else:
                xr, xw, xk, xv, xa, xg = mixed_inputs

            projections = self._tilelang_projections(
                xr, xw, xk, xv, xa, xg, first_value
            )
            if projections is None:
                decay_delta = (xw @ self.w1).tanh() @ self.w2
                decay = self.w0 + decay_delta
                rkv_weight = self._packed_rkv_bmm_weight(xr)
                if rkv_weight is None:
                    receptance = self.receptance(xr)
                    key = self.key(xk)
                    value = self.value(xv)
                else:
                    rkv = torch.bmm(
                        TORCH_STACK((xr, xk, xv), dim=0), rkv_weight
                    )
                    receptance, key, value = rkv.unbind(dim=0)
                value_rank = xv @ self.v1
                a_rank = xa @ self.a1
                g_rank = xg @ self.g1
                if self.layer_id == 0:
                    first_value = value
                else:
                    value = value + (first_value - value) * (
                        self.v0 + value_rank @ self.v2
                    ).sigmoid()
                gate_a = (self.a0 + a_rank @ self.a2).sigmoid()
                gate_g = g_rank.sigmoid() @ self.g2
            else:
                (
                    decay_delta,
                    receptance,
                    key,
                    value,
                    gate_a,
                    gate_g,
                    first_value,
                ) = projections
                decay = self.w0 + decay_delta
        else:
            decay, receptance, key, value, gate_a, gate_g, first_value = (
                graph_outputs
            )

        receptance = receptance.view(batch_size, heads, head_size, 1)
        value = value.view(batch_size, heads, head_size, 1)

        key_gate_outputs = None
        if (
            self.kernel_backend == "tilelang"
            and not self.training
            and not IS_GRAD_ENABLED()
            and batch_size == 1
            and x.dtype in {FLOAT16, BFLOAT16}
            and x.device.type == "cuda"
            and torch.cuda.get_device_capability(x.device) == (12, 0)
        ):
            key_backend, key_workspace = self._tilelang_backend_cache(x)
            key_gate_outputs = key_backend.key_gate(
                key.view(-1),
                self.k_k,
                gate_a.view(-1),
                self.k_a,
                key_workspace,
            )
        if key_gate_outputs is None:
            normalized_key = key * self.k_k
            normalized_key = F.normalize(
                normalized_key.view(batch_size, heads, head_size), dim=-1
            ).view(batch_size, -1)
            key = (key * (1 + (gate_a - 1) * self.k_a)).view(
                batch_size, heads, 1, head_size
            )
        else:
            normalized_key, modified_key, _, _ = key_gate_outputs
            normalized_key = normalized_key.view(batch_size, -1)
            key = modified_key.view(batch_size, heads, 1, head_size)

        tile_mixed = None
        next_matrix = state.time_matrix
        backend = "torch"
        decode_backend = None
        decode_workspace = None
        training_tilelang = (
            self.training
            and self.kernel_backend == "tilelang"
            and x.device.type == "cuda"
            and state.time_matrix.dtype == FLOAT32
        )
        if training_tilelang:
            backend = "tilelang"
        elif not self.training and self.kernel_backend != "torch":
            backend = resolve_backend(self.kernel_backend, x.device)
        if backend == "tilelang" and state.time_matrix.dtype not in {
            FLOAT16,
            BFLOAT16,
            FLOAT32,
        }:
            backend = "torch"
        # FP32 state uses exact TileLang pointwise finalization followed by the
        # native PyTorch projection. SM120 never selects the fused projection.
        if backend == "tilelang":
            try:
                if batch_size == 1 and not training_tilelang:
                    decode_backend, decode_workspace = (
                        self._tilelang_backend_cache(x)
                    )
                use_fused_wkv = (
                    batch_size == 1
                    and x.dtype == FLOAT16
                    and state.time_matrix.dtype == FLOAT16
                    and decay_delta is not None
                    and elapsed_tokens is not None
                    and active is None
                    and decode_backend is not None
                    and decode_workspace is not None
                    and key_gate_outputs is not None
                )
                if use_fused_wkv:
                    decode_backend.wkv_w0_t1(
                        state.time_matrix,
                        receptance.squeeze(-1).view(
                            1, 1, heads, head_size
                        ),
                        decay_delta.view(1, 1, heads, head_size),
                        self.w0.view(heads, head_size),
                        key.squeeze(-2).view(1, 1, heads, head_size),
                        value.squeeze(-1).view(1, 1, heads, head_size),
                        decode_workspace.anti_key.view(
                            1, 1, heads, head_size
                        ),
                        decode_workspace.anti_gate.view(
                            1, 1, heads, head_size
                        ),
                        elapsed_tokens,
                        decode_workspace.wkv_output,
                    )
                    next_matrix = state.time_matrix
                    tile_mixed = decode_workspace.wkv_output.view(
                        1, heads, head_size
                    )
                else:
                    if decay_delta is not None:
                        decay = (-0.606531 * decay.sigmoid()).exp()
                    next_matrix, tile_mixed = tilelang_state_update(
                        state.time_matrix,
                        decay.view(batch_size, heads, head_size),
                        normalized_key.view(batch_size, heads, head_size),
                        gate_a.view(batch_size, heads, head_size),
                        value.squeeze(-1),
                        key.squeeze(-2),
                        receptance.squeeze(-1),
                        state_finalize_op=tilelang_state_finalize_op
                        if training_tilelang
                        else None,
                    )
            except Exception:
                if self.kernel_backend != "auto":
                    raise
                backend = "torch"
        if backend == "torch":
            if decay_delta is not None:
                decay = (-0.606531 * decay.sigmoid()).exp()
            decay = decay.view(batch_size, heads, 1, head_size)
            value_key = value @ key
            anti_value_key = (-normalized_key).view(
                batch_size, heads, head_size, 1
            ) @ (normalized_key * gate_a).view(
                batch_size, heads, 1, head_size
            )
            matrix_f32 = state.time_matrix.float()
            next_matrix = matrix_f32 * decay.to(FLOAT32)
            next_matrix = (
                next_matrix + matrix_f32 @ anti_value_key.to(FLOAT32)
            )
            next_matrix = next_matrix + value_key.to(FLOAT32)
            if state.time_matrix.dtype != FLOAT32:
                next_matrix = next_matrix.to(state.time_matrix.dtype)
        next_shift = x
        if active is not None:
            next_shift = TORCH_WHERE(active, next_shift, state.time_shift)
            next_matrix = TORCH_WHERE(
                active.unsqueeze(-1).unsqueeze(-1),
                next_matrix,
                state.time_matrix,
            )

        projected = (
            tile_mixed
            if tile_mixed is not None
            else (next_matrix.to(dtype=x.dtype) @ receptance).squeeze(-1)
        )
        decode_backend = self._decode_backend
        decode_workspace = self._decode_workspace
        if (
            self.kernel_backend == "tilelang"
            and batch_size == 1
            and decode_backend is not None
            and decode_workspace is not None
            and self.ln_x.weight is not None
            and self.ln_x.bias is not None
        ):
            mixed = decode_backend.post_state(
                projected.view(heads, head_size),
                receptance.squeeze(-1).view(heads, head_size),
                key.squeeze(-2).view(heads, head_size),
                value.squeeze(-1).view(heads, head_size),
                self.r_k,
                gate_g.view(-1),
                self.ln_x.weight,
                self.ln_x.bias,
                float(self.ln_x.eps),
                decode_workspace,
            ).view(1, heads * head_size)
        else:
            mixed = self.ln_x(projected.flatten(start_dim=1))
            rkv = (
                receptance.squeeze(-1) * key.squeeze(-2) * self.r_k
            ).sum(dim=-1, keepdim=True) * value.squeeze(-1)
            mixed = (
                mixed + rkv.view(batch_size, heads * head_size)
            ) * gate_g
        if (
            self.kernel_backend == "tilelang"
            and batch_size == 1
            and decode_backend is not None
            and decode_workspace is not None
        ):
            decode_backend.gemv(
                mixed.view(-1),
                self.output.weight,
                decode_workspace.attention_output,
            )
            attention_output = decode_workspace.attention_output.view(
                1, self.hidden_size
            )
        else:
            attention_output = self.output(mixed)
        return attention_output, first_value, next_shift, next_matrix

    def forward_sequence(
        self,
        x: torch.Tensor,
        state: RWKV7LayerState,
        first_value: torch.Tensor,
        active: torch.Tensor | None,
        reset: torch.Tensor | None,
        tokenwise_projections: bool = False,
        state_scan_backend: str = "torch",
        elapsed_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = x.shape
        heads = self.num_heads
        head_size = self.head_size

        previous, next_shift = _sequence_previous(
            x, state.time_shift, active, reset
        )
        delta = previous - x
        x_mix = self._x_mix_weights(x)
        xr, xw, xk, xv, xa, xg = (
            x.unsqueeze(2) + delta.unsqueeze(2) * x_mix
        ).unbind(dim=2)

        decay_raw = self.w0 + _sequence_matmul(
            _sequence_matmul(xw, self.w1, tokenwise_projections).tanh(),
            self.w2,
            tokenwise_projections,
        )
        decay = (-0.606531 * decay_raw.sigmoid()).exp().view(
            batch_size, sequence_length, heads, head_size
        )
        receptance = _sequence_linear(
            self.receptance, xr, tokenwise_projections
        ).view(batch_size, sequence_length, heads, head_size)
        key = _sequence_linear(self.key, xk, tokenwise_projections)
        value = _sequence_linear(self.value, xv, tokenwise_projections)
        if self.layer_id == 0:
            first_value = value
        else:
            value = value + (first_value - value) * (
                self.v0
                + _sequence_matmul(
                    _sequence_matmul(xv, self.v1, tokenwise_projections),
                    self.v2,
                    tokenwise_projections,
                )
            ).sigmoid()

        value = value.view(batch_size, sequence_length, heads, head_size)
        gate_a = (
            self.a0
            + _sequence_matmul(
                _sequence_matmul(xa, self.a1, tokenwise_projections),
                self.a2,
                tokenwise_projections,
            )
        ).sigmoid()
        gate_g = _sequence_matmul(
            _sequence_matmul(xg, self.g1, tokenwise_projections).sigmoid(),
            self.g2,
            tokenwise_projections,
        )

        normalized_key = key * self.k_k
        normalized_key = F.normalize(
            normalized_key.view(batch_size, sequence_length, heads, head_size),
            dim=-1,
        )
        key = (key * (1 + (gate_a - 1) * self.k_a)).view(
            batch_size, sequence_length, heads, head_size
        )
        gate_a = gate_a.view(batch_size, sequence_length, heads, head_size)

        if state_scan_backend == "tilelang-wkv":
            if active is not None or reset is not None:
                raise RuntimeError(
                    "TileLang WKV prefill does not support masks or resets"
                )
            if elapsed_tokens is None:
                raise RuntimeError("TileLang WKV prefill requires elapsed-token state")
            from .kernel_tilelang_decode import (
                _wkv_precise_out,  # type: ignore[reportMissingImports]
            )

            mixed = torch.empty_like(receptance)
            _wkv_precise_out(
                state.time_matrix,
                receptance,
                decay,
                key,
                value,
                -normalized_key,
                normalized_key * gate_a,
                elapsed_tokens,
                mixed,
            )
            matrix = state.time_matrix
        elif state_scan_backend == "tilelang-fast":
            if active is not None or reset is not None:
                raise RuntimeError(
                    "Fast TileLang sequence scan does not support masks or resets"
                )
            from .kernel_tilelang_state import tilelang_fast_state_scan  # type: ignore[reportMissingImports]

            matrix, mixed = tilelang_fast_state_scan(
                state.time_matrix,
                decay,
                normalized_key,
                gate_a,
                value,
                key,
                receptance,
            )
        elif state_scan_backend == "tilelang":
            from .kernel_tilelang_state import tilelang_state_scan  # type: ignore[reportMissingImports]

            matrix, mixed = tilelang_state_scan(
                state.time_matrix,
                decay,
                normalized_key,
                gate_a,
                value,
                key,
                receptance,
                active=active,
                reset=reset,
                output_mode="full",
            )
        else:
            matrix = state.time_matrix
            mixed_steps: list[torch.Tensor] = []
            for token_index in range(sequence_length):
                if reset is not None:
                    reset_token = reset[:, token_index].unsqueeze(-1).unsqueeze(-1)
                    matrix = TORCH_WHERE(
                        reset_token, TORCH_ZEROS_LIKE(matrix), matrix
                    )
                normalized_token = normalized_key[:, token_index]
                anti_value_key = (-normalized_token).unsqueeze(-1) @ (
                    normalized_token * gate_a[:, token_index]
                ).unsqueeze(-2)
                value_key = value[:, token_index].unsqueeze(-1) @ key[
                    :, token_index
                ].unsqueeze(-2)
                matrix_f32 = matrix.float()
                candidate = (
                    matrix_f32
                    * decay[:, token_index].to(FLOAT32).unsqueeze(-2)
                )
                candidate = (
                    candidate + matrix_f32 @ anti_value_key.to(FLOAT32)
                )
                candidate = candidate + value_key.to(FLOAT32)
                if matrix.dtype != FLOAT32:
                    candidate = candidate.to(matrix.dtype)
                if active is None:
                    matrix = candidate
                else:
                    active_token = active[:, token_index].unsqueeze(-1).unsqueeze(-1)
                    matrix = TORCH_WHERE(active_token, candidate, matrix)
                mixed_steps.append(
                    (
                        matrix.to(dtype=x.dtype)
                        @ receptance[:, token_index].unsqueeze(-1)
                    ).squeeze(-1)
                )
            mixed = TORCH_STACK(mixed_steps, dim=1)
        mixed = self.ln_x(mixed.reshape(batch_size * sequence_length, -1)).view(
            batch_size, sequence_length, heads * head_size
        )
        rkv = (receptance * key * self.r_k).sum(dim=-1, keepdim=True) * value
        mixed = (mixed + rkv.reshape(batch_size, sequence_length, -1)) * gate_g
        return (
            _sequence_linear(self.output, mixed, tokenwise_projections),
            first_value,
            next_shift,
            matrix,
        )


class RWKV7FeedForward(nn.Module):
    def __init__(self, config: RWKV7Config) -> None:
        super().__init__()
        self.x_k = nn.Parameter(TORCH_EMPTY(config.hidden_size))
        self.key = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.value = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.kernel_backend = config.kernel_backend
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_heads = config.num_attention_heads
        self.head_size = config.head_size
        self.ranks = (
            config.decay_lora_rank,
            config.a_lora_rank,
            config.gate_lora_rank,
            config.value_lora_rank,
        )
        self._decode_backend: Any | None = None
        self._decode_workspace: Any | None = None
        self._decode_value_weight: torch.Tensor | None = None
        self._decode_weight_key: tuple[Any, ...] | None = None

    def _clear_inference_cache(self) -> None:
        packed_weight = self._decode_value_weight
        if packed_weight is not None:
            packed_keys = _storage_keys([packed_weight])
            if _storage_keys([self.value.weight]) & packed_keys:
                _restore_parameter_storage(self.value.weight)
        self._decode_backend = None
        self._decode_workspace = None
        self._decode_value_weight = None
        self._decode_weight_key = None

    def _tilelang_supported(self, reference: torch.Tensor) -> bool:
        return (
            self.kernel_backend == "tilelang"
            and not self.training
            and not IS_GRAD_ENABLED()
            and reference.shape == (1, self.hidden_size)
            and reference.dtype in {torch.float16, torch.bfloat16}
            and reference.device.type == "cuda"
            and torch.cuda.get_device_capability(reference.device) == (12, 0)
        )

    def _tilelang_cache(
        self, reference: torch.Tensor
    ) -> tuple[Any, Any, torch.Tensor, bool]:
        from rwkv7_pytorch.tilelang_decode import (
            TileLangDecodeBackend,
            TileLangDecodeSpec,
        )

        weight_key = (
            reference.device,
            reference.dtype,
            getattr(self.key.weight, "_version", 0),
            getattr(self.value.weight, "_version", 0),
        )
        created = (
            self._decode_backend is None
            or self._decode_workspace is None
            or self._decode_value_weight is None
            or self._decode_weight_key != weight_key
        )
        if created:
            self._clear_inference_cache()
            spec = TileLangDecodeSpec(
                channels=self.hidden_size,
                ffn_rows=self.intermediate_size,
                num_heads=self.num_heads,
                head_size=self.head_size,
                ranks=self.ranks,
            )
            backend = TileLangDecodeBackend(
                spec, reference.device, reference.dtype
            )
            packed_value_weight = backend.pack_ffn_value_weight(
                self.value.weight
            )
            _replace_parameter_storage(
                self.value.weight, packed_value_weight.t()
            )
            self._decode_backend = backend
            self._decode_workspace = backend.create_workspace()
            self._decode_value_weight = packed_value_weight
            self._decode_weight_key = (
                reference.device,
                reference.dtype,
                getattr(self.key.weight, "_version", 0),
                getattr(self.value.weight, "_version", 0),
            )
        backend = self._decode_backend
        workspace = self._decode_workspace
        packed_value_weight = self._decode_value_weight
        if (
            backend is None
            or workspace is None
            or packed_value_weight is None
        ):
            raise RuntimeError("TileLang FFN cache initialization failed")
        return backend, workspace, packed_value_weight, created

    def _tilelang_output(self, mixed: torch.Tensor) -> torch.Tensor | None:
        if not self._tilelang_supported(mixed):
            return None
        backend, workspace, packed_value_weight, _ = self._tilelang_cache(mixed)
        backend.ffn(
            mixed.view(-1),
            self.key.weight,
            packed_value_weight,
            workspace.ffn_output,
            workspace,
        )
        return workspace.ffn_output.view(1, self.hidden_size)

    def _tilelang_output_from_residual(
        self,
        residual: torch.Tensor,
        previous: torch.Tensor,
        norm: nn.LayerNorm,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if (
            not self._tilelang_supported(residual)
            or norm.weight is None
            or norm.bias is None
        ):
            return None
        backend, workspace, packed_value_weight, _ = self._tilelang_cache(residual)
        normalized, mixed = backend.cmix_layernorm_mix(
            residual.view(-1),
            previous.view(-1),
            norm.weight,
            norm.bias,
            self.x_k,
            float(norm.eps),
            workspace,
        )
        backend.ffn(
            mixed,
            self.key.weight,
            packed_value_weight,
            workspace.ffn_output,
            workspace,
        )
        return (
            workspace.ffn_output.view(1, self.hidden_size),
            normalized.view(1, self.hidden_size),
        )

    def _tilelang_output_from_residual_update(
        self,
        residual: torch.Tensor,
        update: torch.Tensor,
        previous: torch.Tensor,
        norm: nn.LayerNorm,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if (
            not self._tilelang_supported(residual)
            or update.shape != residual.shape
            or norm.weight is None
            or norm.bias is None
        ):
            return None
        backend, workspace, packed_value_weight, _ = self._tilelang_cache(residual)
        combined, normalized, mixed = backend.cmix_add_layernorm_mix(
            residual.view(-1),
            update.view(-1),
            previous.view(-1),
            norm.weight,
            norm.bias,
            self.x_k,
            float(norm.eps),
            workspace,
        )
        backend.ffn(
            mixed,
            self.key.weight,
            packed_value_weight,
            workspace.ffn_output,
            workspace,
            residual=combined.view(-1),
        )
        return (
            workspace.ffn_output.view(1, self.hidden_size),
            normalized.view(1, self.hidden_size),
            combined.view(1, self.hidden_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        previous: torch.Tensor,
        active: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta = previous - x
        key = x + delta * self.x_k
        output = self._tilelang_output(key)
        if output is None:
            hidden = F.relu(self.key(key)).square()
            output = self.value(hidden)
        next_channel = x
        if active is not None:
            next_channel = TORCH_WHERE(active, next_channel, previous)
        return output, next_channel

    def forward_from_residual(
        self,
        residual: torch.Tensor,
        previous: torch.Tensor,
        norm: nn.LayerNorm,
        active: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tilelang_output = self._tilelang_output_from_residual(
            residual, previous, norm
        )
        if tilelang_output is None:
            return self.forward(norm(residual), previous, active)
        output, next_channel = tilelang_output
        if active is not None:
            next_channel = TORCH_WHERE(active, next_channel, previous)
        return output, next_channel

    def forward_from_residual_update(
        self,
        residual: torch.Tensor,
        update: torch.Tensor,
        previous: torch.Tensor,
        norm: nn.LayerNorm,
        active: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tilelang_output = (
            None
            if active is not None
            else self._tilelang_output_from_residual_update(
                residual, update, previous, norm
            )
        )
        if tilelang_output is None:
            combined = residual + update
            output, next_channel = self.forward_from_residual(
                combined, previous, norm, active
            )
            return combined + output, next_channel, combined
        output, next_channel, combined = tilelang_output
        return output, next_channel, combined

    def forward_sequence(
        self,
        x: torch.Tensor,
        previous: torch.Tensor,
        active: torch.Tensor | None,
        reset: torch.Tensor | None,
        tokenwise_projections: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        previous_sequence, next_channel = _sequence_previous(
            x, previous, active, reset
        )
        key = x + (previous_sequence - x) * self.x_k
        hidden = F.relu(
            _sequence_linear(self.key, key, tokenwise_projections)
        ).square()
        output = _sequence_linear(self.value, hidden, tokenwise_projections)
        return output, next_channel


class RWKV7Block(nn.Module):
    def __init__(self, config: RWKV7Config, layer_id: int) -> None:
        super().__init__()
        hidden = config.hidden_size
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(hidden, eps=config.layer_norm_epsilon)
        self.ln1 = nn.LayerNorm(hidden, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(hidden, eps=config.layer_norm_epsilon)
        self.att = RWKV7Attention(config, layer_id)
        self.ffn = RWKV7FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        state: RWKV7LayerState,
        first_value: torch.Tensor,
        active: torch.Tensor | None,
        elapsed_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RWKV7LayerState, torch.Tensor]:
        time_mix = self.att._tilelang_layernorm_mix6(
            x, state.time_shift, self.ln1, active
        )
        if time_mix is None:
            normalized = self.ln1(x)
            mixed_inputs = None
        else:
            normalized, mixed_inputs = time_mix
        mixed, first_value, time_shift, time_matrix = self.att(
            normalized,
            state,
            first_value,
            active,
            elapsed_tokens,
            mixed_inputs,
        )
        x, channel, _ = self.ffn.forward_from_residual_update(
            x, mixed, state.channel, self.ln2, active
        )
        return x, RWKV7LayerState(channel, time_shift, time_matrix), first_value


    def forward_sequence(
        self,
        x: torch.Tensor,
        state: RWKV7LayerState,
        first_value: torch.Tensor,
        active: torch.Tensor | None,
        reset: torch.Tensor | None,
        tokenwise_projections: bool = False,
        state_scan_backend: str = "torch",
        elapsed_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RWKV7LayerState, torch.Tensor]:
        mixed, first_value, time_shift, time_matrix = self.att.forward_sequence(
            self.ln1(x),
            state,
            first_value,
            active,
            reset,
            tokenwise_projections,
            state_scan_backend,
            elapsed_tokens,
        )
        x = x + mixed
        mixed, channel = self.ffn.forward_sequence(
            self.ln2(x),
            state.channel,
            active,
            reset,
            tokenwise_projections,
        )
        x = x + mixed
        return x, RWKV7LayerState(channel, time_shift, time_matrix), first_value


class RWKV7ForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = RWKV7Config
    base_model_prefix = "rwkv7"
    main_input_name = "input_ids"
    _is_stateful = True
    _supports_cache_class = True
    supports_gradient_checkpointing = True
    _supports_sdpa = False
    _supports_flash_attn = False

    def __init__(self, config: RWKV7Config) -> None:
        super().__init__(config)
        self.emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            [
                RWKV7Block(config, layer_id)
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.ln_out = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.kernel_backend = config.kernel_backend
        self._inference_cache_epoch = 0
        self.gradient_checkpointing = False
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        for name, parameter in module.named_parameters(recurse=False):
            if name not in {"weight", "bias"}:
                nn.init.zeros_(parameter)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.emb

    def set_input_embeddings(self, value: nn.Module) -> None:
        if not isinstance(value, nn.Embedding):
            raise TypeError("Input embeddings must be nn.Embedding")
        self.emb = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        if not isinstance(new_embeddings, nn.Linear):
            raise TypeError("Output embeddings must be nn.Linear")
        self.head = new_embeddings

    @property
    def inference_cache_epoch(self) -> int:
        """Generation counter used to invalidate captured inference runners."""
        return self._inference_cache_epoch

    def inference_cache_stats(self) -> dict[str, int]:
        """Return bounded tensor/cache counts without allocating new device storage."""
        attention_values: list[Any] = []
        mix_values: list[Any] = []
        ffn_values: list[Any] = []
        workspace_values: list[Any] = []
        backend_count = 0
        workspace_count = 0
        time_mix_graph_count = 0
        for block in cast(Any, self.blocks):
            mix_values.append(block.att._x_mix_cache)
            attention_values.extend(
                (
                    block.att._rkv_bmm_weight,
                    block.att._decode_weight,
                    block.att._decode_rankout_weights,
                )
            )
            ffn_values.append(block.ffn._decode_value_weight)
            workspace_values.extend(
                (block.att._decode_workspace, block.ffn._decode_workspace)
            )
            backend_count += int(block.att._decode_backend is not None)
            backend_count += int(block.ffn._decode_backend is not None)
            workspace_count += int(block.att._decode_workspace is not None)
            workspace_count += int(block.ffn._decode_workspace is not None)
            time_mix_graph_count += int(block.att._time_mix_graph is not None)
        packed_values = attention_values + ffn_values
        parameter_values: list[Any] = list(self.parameters())
        parameter_keys = _storage_keys(parameter_values)
        packed_bytes = _unique_storage_bytes(packed_values)
        extra_packed_bytes = _unique_storage_bytes_excluding(
            packed_values, parameter_keys
        )
        workspace_bytes = _unique_storage_bytes(workspace_values)
        all_cache_values = packed_values + mix_values + workspace_values
        return {
            "parameter_storage_bytes": _unique_storage_bytes(parameter_values),
            "mix_cache_bytes": _unique_storage_bytes(mix_values),
            "packed_attention_bytes": _unique_storage_bytes(attention_values),
            "packed_ffn_bytes": _unique_storage_bytes(ffn_values),
            "packed_weight_bytes": packed_bytes,
            "shared_layout_bytes": packed_bytes - extra_packed_bytes,
            "extra_packed_weight_bytes": extra_packed_bytes,
            "workspace_bytes": workspace_bytes,
            "extra_tensor_bytes": _unique_storage_bytes_excluding(
                all_cache_values, parameter_keys
            ),
            "total_tensor_bytes": _unique_storage_bytes(
                all_cache_values
            ),
            "backend_count": backend_count,
            "workspace_count": workspace_count,
            "time_mix_graph_count": time_mix_graph_count,
            "epoch": self._inference_cache_epoch,
        }

    def clear_inference_caches(self, *, include_compiled: bool = False) -> None:
        """Release model-owned inference layouts, workspaces, and graph wrappers.

        Captured runners record the cache epoch and refuse replay after this call.
        Global compiled TileLang kernels remain cached unless ``include_compiled`` is
        requested explicitly.
        """
        for block in cast(Any, self.blocks):
            attention = block.att
            attention._x_mix_cache = None
            attention._x_mix_cache_versions = None
            attention._time_mix_graph = None
            attention._time_mix_graph_key = None
            attention._clear_inference_cache()
            attention._rkv_bmm_weight = None
            attention._rkv_bmm_weight_key = None
            feed_forward = block.ffn
            feed_forward._clear_inference_cache()
        self._inference_cache_epoch += 1
        if include_compiled:
            from .kernel_tilelang_decode import clear_tilelang_kernel_caches
            from .kernel_tilelang_state import clear_tilelang_state_kernel_caches
            from .tilelang_decode import clear_tilelang_runtime_caches

            clear_tilelang_kernel_caches()
            clear_tilelang_state_kernel_caches()
            clear_tilelang_runtime_caches()

    def prepare_inference_weights(self) -> dict[str, int]:
        """Install shared TileLang layouts without duplicating model parameters.

        The canonical parameters become views of the read-only inference layouts.
        ``clear_inference_caches`` restores independent contiguous parameters before
        training, device moves, state-dict serialization, or backend changes.
        """
        if self.kernel_backend != "tilelang":
            raise RuntimeError("inference layouts require explicit TileLang backend")
        if self.training:
            raise RuntimeError("inference layouts require eval mode")
        reference = self.emb.weight
        if (
            reference.device.type != "cuda"
            or reference.dtype not in {FLOAT16, BFLOAT16}
            or torch.cuda.get_device_capability(reference.device) != (12, 0)
        ):
            raise RuntimeError(
                "inference layouts require FP16/BF16 parameters on SM120"
            )
        created = False
        with torch.inference_mode():
            sample = reference.new_empty((1, self.config.hidden_size))
            for block in cast(Any, self.blocks):
                *_, attention_created = block.att._prepare_tilelang_cache(sample)
                *_, ffn_created = block.ffn._tilelang_cache(sample)
                created = created or attention_created or ffn_created
        if created:
            self._inference_cache_epoch += 1
        return self.inference_cache_stats()

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if hasattr(self, "_inference_cache_epoch"):
            self.clear_inference_caches()
        return cast(dict[str, Any], super().state_dict(*args, **kwargs))

    def load_state_dict(self, *args: Any, **kwargs: Any) -> Any:
        if hasattr(self, "_inference_cache_epoch"):
            self.clear_inference_caches()
        return super().load_state_dict(*args, **kwargs)

    def set_kernel_backend(self, backend: str) -> None:
        if backend not in {"auto", "torch", "tilelang"}:
            raise ValueError(f"Unsupported kernel backend: {backend}")
        if backend != self.kernel_backend:
            self.clear_inference_caches()
        self.kernel_backend = backend
        self.config.kernel_backend = backend
        for block in self.blocks:
            if isinstance(block, RWKV7Block):
                block.att.kernel_backend = backend
                block.ffn.kernel_backend = backend

    def train(self, mode: bool = True) -> "RWKV7ForCausalLM":
        if mode and hasattr(self, "_inference_cache_epoch"):
            self.clear_inference_caches()
        return cast("RWKV7ForCausalLM", super().train(mode))

    def _apply(self, fn: Any, recurse: bool = True) -> "RWKV7ForCausalLM":
        if hasattr(self, "_inference_cache_epoch"):
            self.clear_inference_caches()
        return cast("RWKV7ForCausalLM", super()._apply(fn, recurse=recurse))


    def set_cuda_graph_time_mix(self, enabled: bool) -> None:
        """Enable exact warm CUDA-graph replay for one-token time-mix inputs."""
        if enabled:
            dynamo_config = __import__(
                "torch._dynamo.config", fromlist=["cache_size_limit"]
            )
            minimum_limit = max(64, len(self.blocks) * 2)
            for name in ("cache_size_limit", "recompile_limit"):
                current = getattr(dynamo_config, name, 0)
                if current < minimum_limit:
                    setattr(dynamo_config, name, minimum_limit)
        for block in cast(Any, self.blocks):
            block.att.set_cuda_graph_time_mix(enabled)


    def init_state(self, batch_size: int, device, dtype) -> RWKV7State:
        matrix_dtype = {
            "float32": FLOAT32,
            "float16": FLOAT16,
            "bfloat16": BFLOAT16,
        }[self.config.recurrent_state_dtype]
        return RWKV7State.empty(
            num_layers=self.config.num_hidden_layers,
            batch_size=batch_size,
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            head_size=self.config.head_size,
            device=device,
            dtype=dtype,
            matrix_dtype=matrix_dtype,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        next_sequence_length: int | None = None,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        is_first_iteration: bool | None = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del next_sequence_length, cache_position, is_first_iteration
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            inputs_embeds = None
        model_inputs: dict[str, Any] = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": kwargs.pop("use_cache", True),
            **kwargs,
        }
        if inputs_embeds is not None and past_key_values is None:
            model_inputs["inputs_embeds"] = inputs_embeds
            model_inputs["input_ids"] = None
        return model_inputs

    def _forward_token_sequence(
        self,
        embeds: torch.Tensor,
        state: RWKV7State,
        sequence_mask: torch.Tensor | None,
        reset_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, RWKV7State]:
        sequence_length = embeds.shape[1]
        hidden_steps: list[torch.Tensor] = []
        for token_index in range(sequence_length):
            x = embeds[:, token_index]
            active = None
            if sequence_mask is not None:
                active = sequence_mask[:, token_index].unsqueeze(-1)
            reset = None
            if reset_mask is not None:
                reset = reset_mask[:, token_index].unsqueeze(-1)
            _reset_elapsed_tokens(state.elapsed_tokens, reset)
            first_block = self.blocks[0]
            if not isinstance(first_block, RWKV7Block):
                raise TypeError("Invalid first RWKV block")
            x = first_block.ln0(x)
            first_value = x
            next_layers: list[RWKV7LayerState] = []
            for block, layer_state in zip(
                self.blocks, state.layer_states, strict=True
            ):
                if not isinstance(block, RWKV7Block):
                    raise TypeError("Invalid RWKV block")
                layer_state = _reset_layer_state(layer_state, reset)
                if self.gradient_checkpointing and self.training:
                    checkpoint = self._gradient_checkpointing_func

                    def block_step(
                        hidden: torch.Tensor,
                        channel: torch.Tensor,
                        time_shift: torch.Tensor,
                        time_matrix: torch.Tensor,
                        first: torch.Tensor,
                        *,
                        current_block: RWKV7Block = block,
                    ) -> tuple[
                        torch.Tensor,
                        torch.Tensor,
                        torch.Tensor,
                        torch.Tensor,
                        torch.Tensor,
                    ]:
                        next_hidden, current_state, next_first = current_block(
                            hidden,
                            RWKV7LayerState(channel, time_shift, time_matrix),
                            first,
                            active,
                            state.elapsed_tokens,
                        )
                        return (
                            next_hidden,
                            current_state.channel,
                            current_state.time_shift,
                            current_state.time_matrix,
                            next_first,
                        )

                    x, channel, time_shift, time_matrix, first_value = checkpoint(
                        block_step,
                        x,
                        layer_state.channel,
                        layer_state.time_shift,
                        layer_state.time_matrix,
                        first_value,
                    )
                    next_state = RWKV7LayerState(
                        channel, time_shift, time_matrix
                    )
                else:
                    x, next_state, first_value = block(
                        x,
                        layer_state,
                        first_value,
                        active,
                        state.elapsed_tokens,
                    )
                next_layers.append(next_state)
            _advance_elapsed_tokens(state.elapsed_tokens, active)
            state = RWKV7State(
                next_layers,
                state.seen_tokens + 1,
                state.elapsed_tokens,
            )
            hidden_steps.append(self.ln_out(x))
        return TORCH_STACK(hidden_steps, dim=1), state

    def _forward_layer_sequence(
        self,
        embeds: torch.Tensor,
        state: RWKV7State,
        sequence_mask: torch.Tensor | None,
        reset_mask: torch.Tensor | None,
        tokenwise_projections: bool,
        state_scan_backend: str = "torch",
    ) -> tuple[torch.Tensor, RWKV7State]:
        first_block = self.blocks[0]
        if not isinstance(first_block, RWKV7Block):
            raise TypeError("Invalid first RWKV block")
        x = first_block.ln0(embeds)
        first_value = x
        active = None if sequence_mask is None else sequence_mask.unsqueeze(-1)
        reset = None if reset_mask is None else reset_mask.unsqueeze(-1)
        next_layers: list[RWKV7LayerState] = []
        for block, layer_state in zip(self.blocks, state.layer_states, strict=True):
            if not isinstance(block, RWKV7Block):
                raise TypeError("Invalid RWKV block")
            if self.gradient_checkpointing and self.training:
                checkpoint = self._gradient_checkpointing_func

                def block_sequence(
                    hidden: torch.Tensor,
                    channel: torch.Tensor,
                    time_shift: torch.Tensor,
                    time_matrix: torch.Tensor,
                    first: torch.Tensor,
                    *,
                    current_block: RWKV7Block = block,
                ) -> tuple[
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                ]:
                    next_hidden, current_state, next_first = (
                        current_block.forward_sequence(
                            hidden,
                            RWKV7LayerState(channel, time_shift, time_matrix),
                            first,
                            active,
                            reset,
                            tokenwise_projections,
                            state_scan_backend,
                            state.elapsed_tokens,
                        )
                    )
                    return (
                        next_hidden,
                        current_state.channel,
                        current_state.time_shift,
                        current_state.time_matrix,
                        next_first,
                    )

                x, channel, time_shift, time_matrix, first_value = checkpoint(
                    block_sequence,
                    x,
                    layer_state.channel,
                    layer_state.time_shift,
                    layer_state.time_matrix,
                    first_value,
                )
                next_state = RWKV7LayerState(channel, time_shift, time_matrix)
            else:
                x, next_state, first_value = block.forward_sequence(
                    x,
                    layer_state,
                    first_value,
                    active,
                    reset,
                    tokenwise_projections,
                    state_scan_backend,
                    state.elapsed_tokens,
                )
            next_layers.append(next_state)
        elapsed_tokens = state.elapsed_tokens
        for token_index in range(embeds.shape[1]):
            token_active = None if active is None else active[:, token_index]
            token_reset = None if reset is None else reset[:, token_index]
            _reset_elapsed_tokens(elapsed_tokens, token_reset)
            _advance_elapsed_tokens(elapsed_tokens, token_active)
        next_state = RWKV7State(
            next_layers,
            state.seen_tokens + embeds.shape[1],
            elapsed_tokens,
        )
        return self.ln_out(x), next_state

    def _forward_chunked_layer_sequence(
        self,
        embeds: torch.Tensor,
        state: RWKV7State,
        sequence_mask: torch.Tensor | None,
        reset_mask: torch.Tensor | None,
        *,
        chunk_size: int,
        tokenwise_projections: bool,
        state_scan_backend: str,
        hidden_to_keep: int | None,
    ) -> tuple[torch.Tensor, RWKV7State]:
        hidden_chunks: list[torch.Tensor] = []
        retained_hidden: torch.Tensor | None = None
        for start in range(0, embeds.shape[1], chunk_size):
            stop = min(start + chunk_size, embeds.shape[1])
            chunk_mask = (
                None if sequence_mask is None else sequence_mask[:, start:stop]
            )
            chunk_reset = (
                None if reset_mask is None else reset_mask[:, start:stop]
            )
            chunk_hidden, state = self._forward_layer_sequence(
                embeds[:, start:stop],
                state,
                chunk_mask,
                chunk_reset,
                tokenwise_projections=tokenwise_projections,
                state_scan_backend=state_scan_backend,
            )
            if hidden_to_keep is None:
                hidden_chunks.append(chunk_hidden)
            elif retained_hidden is None:
                retained_hidden = chunk_hidden[:, -hidden_to_keep:]
            else:
                retained_hidden = TORCH_CAT(
                    (retained_hidden, chunk_hidden), dim=1
                )[:, -hidden_to_keep:]
        if hidden_to_keep is not None:
            if retained_hidden is None:
                raise RuntimeError("Chunked prefill produced no hidden states")
            return retained_hidden, state
        return TORCH_CAT(hidden_chunks, dim=1), state

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        past_key_values: RWKV7State | None = None,
        use_cache: bool | None = None,
        inputs_embeds: torch.Tensor | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        state_reset_mask: torch.Tensor | None = None,
        sequence_mode: str = "auto",
        rwkv_prefill_chunk_size: int | None = None,
        **kwargs: Any,
    ) -> Any:
        if "prefill_chunk_size" in kwargs:
            raise TypeError(
                "Use rwkv_prefill_chunk_size; prefill_chunk_size is reserved by "
                "Transformers generation"
            )
        del kwargs
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Pass exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids is required when inputs_embeds is absent")
            embeds = self.emb(input_ids)
        else:
            embeds = inputs_embeds
        if embeds.ndim != 3:
            raise ValueError(
                f"Expected [batch, sequence, hidden], got {tuple(embeds.shape)}"
            )

        batch_size, sequence_length, _ = embeds.shape
        if sequence_length == 0:
            raise ValueError("RWKV requires at least one input token")
        if past_key_values is None:
            state = self.init_state(batch_size, embeds.device, embeds.dtype)
        elif not isinstance(past_key_values, RWKV7State):
            raise TypeError("past_key_values must be RWKV7State")
        else:
            state = past_key_values
        if state.max_batch_size != batch_size:
            raise ValueError("RWKV state batch size does not match input batch")

        if attention_mask is not None:
            attention_mask = attention_mask[:, -sequence_length:].to(dtype=BOOL)
        sequence_mask = attention_mask
        if state_reset_mask is not None:
            state_reset_mask = state_reset_mask[:, -sequence_length:].to(dtype=BOOL)
        if sequence_mode not in {
            "auto",
            "token",
            "layer",
            "layer-exact",
            "tilelang-scan",
            "tilelang-scan-fast",
        }:
            raise ValueError(
                "sequence_mode must be auto, token, layer, layer-exact, "
                "tilelang-scan or tilelang-scan-fast"
            )
        use_tilelang_prefill = (
            sequence_mode == "auto"
            and sequence_length > 1
            and not self.training
            and not IS_GRAD_ENABLED()
            and sequence_mask is None
            and state_reset_mask is None
            and batch_size == 1
            and embeds.dtype == FLOAT16
            and embeds.device.type == "cuda"
            and torch.cuda.get_device_capability(embeds.device) == (12, 0)
            and state.elapsed_tokens is not None
            and state.elapsed_tokens.dtype == torch.int32
            and state.elapsed_tokens.is_contiguous()
            and all(
                layer.time_matrix.dtype == FLOAT16
                and layer.time_matrix.is_contiguous()
                for layer in state.layer_states
            )
            and self.kernel_backend == "tilelang"
        )
        if sequence_mode == "auto":
            resolved_mode = (
                "token"
                if sequence_length == 1
                else "tilelang-prefill"
                if use_tilelang_prefill
                else "layer-exact"
            )
        else:
            resolved_mode = sequence_mode
        if resolved_mode in {
            "tilelang-scan",
            "tilelang-scan-fast",
        } and self.training:
            raise RuntimeError("TileLang sequence scan is inference-only")
        tokenwise_projections = resolved_mode in {
            "layer-exact",
            "tilelang-scan",
            "tilelang-scan-fast",
        }
        state_scan_backend = (
            "tilelang-wkv"
            if resolved_mode == "tilelang-prefill"
            else "tilelang-fast"
            if resolved_mode == "tilelang-scan-fast"
            else "tilelang"
            if resolved_mode == "tilelang-scan"
            else "torch"
        )

        requested_chunk_size = (
            0
            if rwkv_prefill_chunk_size is None and self.training
            else self.config.rwkv_prefill_chunk_size
            if rwkv_prefill_chunk_size is None
            else rwkv_prefill_chunk_size
        )
        if requested_chunk_size < 0:
            raise ValueError("rwkv_prefill_chunk_size must be non-negative")
        hidden_to_keep = None
        if (
            labels is None
            and isinstance(logits_to_keep, int)
            and logits_to_keep > 0
        ):
            hidden_to_keep = min(logits_to_keep, sequence_length)

        if resolved_mode == "token":
            hidden, state = self._forward_token_sequence(
                embeds, state, sequence_mask, state_reset_mask
            )
        elif 0 < requested_chunk_size < sequence_length:
            hidden, state = self._forward_chunked_layer_sequence(
                embeds,
                state,
                sequence_mask,
                state_reset_mask,
                chunk_size=requested_chunk_size,
                tokenwise_projections=tokenwise_projections,
                state_scan_backend=state_scan_backend,
                hidden_to_keep=hidden_to_keep,
            )
        else:
            hidden, state = self._forward_layer_sequence(
                embeds,
                state,
                sequence_mask,
                state_reset_mask,
                tokenwise_projections=tokenwise_projections,
                state_scan_backend=state_scan_backend,
            )

        loss: Any = None
        head_input = hidden
        if labels is None:
            if isinstance(logits_to_keep, int) and logits_to_keep > 0:
                head_input = hidden[:, -logits_to_keep:]
            elif isinstance(logits_to_keep, torch.Tensor):
                head_input = hidden.index_select(
                    1, logits_to_keep.to(device=hidden.device)
                )
        logits = self.head(head_input)
        if labels is not None:
            if labels.shape != logits.shape[:2]:
                raise ValueError("labels must match input_ids shape")
            effective_labels = labels
            if sequence_mask is not None:
                effective_labels = labels.masked_fill(sequence_mask == 0, -100)
            if state_reset_mask is not None:
                effective_labels = effective_labels.masked_fill(
                    state_reset_mask, -100
                )
            shift_logits = logits[:, :-1].contiguous().float()
            shift_labels = effective_labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_state = state if use_cache else None
        return_dict = (
            bool(getattr(self.config, "return_dict", True))
            if return_dict is None
            else return_dict
        )
        if not return_dict:
            output = (logits, output_state)
            return ((loss,) + output) if loss is not None else output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=output_state,
        )
