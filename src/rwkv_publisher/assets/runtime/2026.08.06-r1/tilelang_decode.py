from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .kernel_tilelang_decode import (
    _compiled_cmix_add_layernorm_mix,
    _compiled_cmix_finalize,
    _compiled_cmix_layernorm_mix,
    _compiled_cmix_value,
    _compiled_ffn,
    _compiled_gemv,
    _compiled_cmix_binned_finalize,
    _compiled_cmix_sparse_binned,
    _compiled_key_gate,
    _compiled_post_state,
    _compiled_tmix_layernorm_mix6,
    _compiled_rankout,
    _compiled_rankout_reduced,
    _compiled_rkv,
    _wkv_out,
    _wkv_w0_t1_out,
)
from .kernel_tilelang_state import cuda_arch_key

_STREAM_POOLS: dict[int, tuple[torch.cuda.Stream, ...]] = {}


def _stream_pool(device: torch.device) -> tuple[torch.cuda.Stream, ...]:
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    streams = _STREAM_POOLS.get(index)
    if streams is None:
        with torch.cuda.device(index):
            streams = tuple(torch.cuda.Stream() for _ in range(4))
        _STREAM_POOLS[index] = streams
    return streams


@dataclass(frozen=True)
class TileLangDecodeSpec:
    """Fixed B1T1 dimensions for optimized inference."""

    channels: int
    ffn_rows: int
    num_heads: int
    head_size: int
    ranks: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if self.channels <= 0 or self.ffn_rows <= 0:
            raise ValueError("decode dimensions must be positive")
        if self.num_heads * self.head_size != self.channels:
            raise ValueError("num_heads * head_size must equal channels")
        if self.head_size != 64:
            raise ValueError("TileLang WKV requires head size 64")
        if len(self.ranks) != 4 or any(rank < 0 for rank in self.ranks):
            raise ValueError("ranks must contain four non-negative values")

    @property
    def rkv_output_rows(self) -> int:
        return 3 * self.channels + sum(self.ranks)


@dataclass
class TileLangDecodeWorkspace:
    """Caller-owned graph-capturable workspace for one layer/request."""

    rkv_output: torch.Tensor
    rank_decay: torch.Tensor
    rank_gate_a: torch.Tensor
    rank_gate_g: torch.Tensor
    rank_value: torch.Tensor
    wkv_output: torch.Tensor
    attention_output: torch.Tensor
    normalized_key: torch.Tensor
    modified_key: torch.Tensor
    anti_key: torch.Tensor
    anti_gate: torch.Tensor
    block_residual: torch.Tensor
    tmix_normalized: torch.Tensor
    tmix_mixed: torch.Tensor
    post_mixed: torch.Tensor
    ffn_normalized: torch.Tensor
    ffn_hidden: torch.Tensor
    ffn_partials: torch.Tensor
    ffn_output: torch.Tensor
    ffn_bins: torch.Tensor



class TileLangDecodeBackend:
    """Single ultra-optimized SM120 TileLang B1T1 inference backend.

    PyTorch remains pure reference implementation. Callers own packed weights,
    recurrent state, and workspaces; backend performs no hidden allocation.
    """

    def __init__(
        self,
        spec: TileLangDecodeSpec,
        device: torch.device | str,
        dtype: torch.dtype = torch.float16,
    ):
        self.spec = spec
        requested_device = torch.device(device)
        self.device = (
            torch.device("cuda", torch.cuda.current_device())
            if requested_device.type == "cuda" and requested_device.index is None
            else requested_device
        )
        if self.device.type != "cuda":
            raise RuntimeError("TileLang decode backend requires CUDA")
        if torch.cuda.get_device_capability(self.device) != (12, 0):
            raise RuntimeError("TileLang decode backend requires SM120")
        if dtype not in {torch.float16, torch.bfloat16}:
            raise TypeError("TileLang decode backend requires FP16 or BF16")
        self.dtype = dtype
        self.input_dtype = "float16" if dtype == torch.float16 else "bfloat16"
        self._architecture = cuda_arch_key(self.device)
        self._rkv_kernel: Any | None = None
        self._gemv_kernels: dict[tuple[int, int, bool], Any] = {}
        self._ffn_kernel: Any | None = None
        self._cmix_sparse_binned_kernel: Any | None = None
        self._cmix_binned_finalize_kernel: Any | None = None
        self._cmix_layernorm_mix_kernel: Any | None = None
        self._cmix_add_layernorm_mix_kernel: Any | None = None
        self._tmix_layernorm_mix6_kernel: Any | None = None
        self._key_gate_kernel: Any | None = None
        self._cmix_value_kernel: Any | None = None
        self._cmix_finalize_kernel: Any | None = None
        self._rkv_binding: tuple[int, ...] | None = None
        self._ffn_binding: tuple[int, ...] | None = None
        self._rankout_kernels: dict[bool, Any] = {}
        self._rankout_bindings: dict[bool, tuple[int, ...]] = {}
        self._rankout_reduced_kernels: dict[bool, Any] = {}
        self._rankout_reduced_bindings: dict[bool, tuple[int, ...]] = {}
        self._post_state_kernel: Any | None = None
        self._post_state_binding: tuple[int, ...] | None = None
        self._streams = _stream_pool(self.device)

    def create_workspace(self) -> TileLangDecodeWorkspace:
        options = {"device": self.device, "dtype": self.dtype}
        return TileLangDecodeWorkspace(
            rkv_output=torch.empty(self.spec.rkv_output_rows, **options),
            rank_decay=torch.empty(self.spec.channels, **options),
            rank_gate_a=torch.empty(self.spec.channels, **options),
            rank_gate_g=torch.empty(self.spec.channels, **options),
            rank_value=torch.empty(self.spec.channels, **options),
            wkv_output=torch.empty(
                (1, 1, self.spec.num_heads, self.spec.head_size), **options
            ),
            attention_output=torch.empty(self.spec.channels, **options),
            normalized_key=torch.empty(self.spec.channels, **options),
            modified_key=torch.empty(self.spec.channels, **options),
            anti_key=torch.empty(self.spec.channels, **options),
            anti_gate=torch.empty(self.spec.channels, **options),
            block_residual=torch.empty(self.spec.channels, **options),
            tmix_normalized=torch.empty(self.spec.channels, **options),
            tmix_mixed=torch.empty((6, self.spec.channels), **options),
            post_mixed=torch.empty(self.spec.channels, **options),
            ffn_normalized=torch.empty(self.spec.channels, **options),
            ffn_hidden=torch.empty(self.spec.ffn_rows, **options),
            ffn_partials=torch.empty(
                (4, self.spec.channels), **options
            ),
            ffn_output=torch.empty(self.spec.channels, **options),
            ffn_bins=torch.zeros(
                (6, self.spec.channels),
                device=self.device,
                dtype=torch.float32,
            ),
        )

    def pack_rkv_weights(
        self,
        rkv_weights: Sequence[torch.Tensor],
        lowrank_weights: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Pack three dense and four low-rank weights once during model load."""
        if len(rkv_weights) != 3 or len(lowrank_weights) != 4:
            raise ValueError("expected three RKV and four low-rank weights")
        expected_rows = (self.spec.channels,) * 3 + self.spec.ranks
        weights = tuple(rkv_weights) + tuple(lowrank_weights)
        for weight, rows in zip(weights, expected_rows, strict=True):
            self._validate(weight, (rows, self.spec.channels), "weight")
        return torch.cat(weights, dim=0).contiguous()

    def pack_ffn_value_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Pack FFN value weight as activation-major contiguous tiles."""
        self._validate(
            weight,
            (self.spec.channels, self.spec.ffn_rows),
            "FFN value weight",
        )
        return weight.t().contiguous()

    def rkv(
        self,
        inputs: Sequence[torch.Tensor],
        packed_weight: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Run direct-input R/K/V and W/A/G/V rank-input projection."""
        if len(inputs) != 6:
            raise ValueError("expected xr, xk, xv, xw, xa, and xg")
        tensors = (*inputs, packed_weight, output)
        self._reject_training(tensors)
        if self._rkv_kernel is None:
            self._rkv_kernel = _compiled_rkv(
                self.spec.channels,
                *self.spec.ranks,
                2,
                128,
                self.input_dtype,
                self._architecture,
            )
        binding = tuple(tensor.data_ptr() for tensor in tensors)
        if binding != self._rkv_binding:
            for value in inputs:
                self._validate(value, (self.spec.channels,), "RKV input")
            self._validate(
                packed_weight,
                (self.spec.rkv_output_rows, self.spec.channels),
                "packed RKV weight",
            )
            self._validate(
                output, (self.spec.rkv_output_rows,), "RKV output"
            )
            self._rkv_binding = binding
        self._rkv_kernel(*inputs, packed_weight, output)

    def gemv(
        self,
        value: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
        clear_output: torch.Tensor | None = None,
    ) -> None:
        """Run output-tiled inference GEMV into caller-owned output."""
        input_rows = value.numel()
        output_rows = output.numel()
        clear_target = value if clear_output is None else clear_output
        tensors = (value, weight, output, clear_target)
        self._reject_training(tensors)
        key = (input_rows, output_rows, clear_output is not None)
        kernel = self._gemv_kernels.get(key)
        if kernel is None:
            kernel = _compiled_gemv(
                input_rows,
                output_rows,
                self.input_dtype,
                2,
                128,
                clear_output is not None,
                self._architecture,
            )
            self._gemv_kernels[key] = kernel
        self._validate(value, (input_rows,), "GEMV input")
        self._validate(weight, (output_rows, input_rows), "GEMV weight")
        self._validate(output, (output_rows,), "GEMV output")
        self._validate(clear_target, (input_rows,), "GEMV clear output")
        kernel(*tensors)

    def rkv_views(self, output: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return zero-copy R, K, V, W1, A1, G1, V1 views."""
        sizes = (self.spec.channels,) * 3 + self.spec.ranks
        return tuple(output.split(sizes))

    def rankout(
        self,
        ranks: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        vectors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        value_base: torch.Tensor,
        first_value: torch.Tensor,
        workspace: TileLangDecodeWorkspace,
        *,
        use_value_mix: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fuse W/A/G/V rank-out and pointwise finalization."""
        outputs = (
            workspace.rank_decay,
            workspace.rank_gate_a,
            workspace.rank_gate_g,
            workspace.rank_value,
        )
        tensors = (*ranks, *weights, *vectors, value_base, first_value, *outputs)
        self._reject_training(tensors)
        kernel = self._rankout_kernels.get(use_value_mix)
        if kernel is None:
            kernel = _compiled_rankout(
                self.spec.channels,
                *self.spec.ranks,
                use_value_mix,
                self.input_dtype,
                self._architecture,
            )
            self._rankout_kernels[use_value_mix] = kernel
        binding = tuple(tensor.data_ptr() for tensor in tensors)
        if self._rankout_bindings.get(use_value_mix) != binding:
            for rank, size in zip(ranks, self.spec.ranks, strict=True):
                self._validate(rank, (size,), "rank-out input")
            for weight, size in zip(weights, self.spec.ranks, strict=True):
                self._validate(
                    weight, (size, self.spec.channels), "rank-out weight"
                )
            for vector in vectors:
                self._validate(
                    vector, (self.spec.channels,), "rank-out vector"
                )
            self._validate(
                value_base, (self.spec.channels,), "rank-out value"
            )
            self._validate(
                first_value, (self.spec.channels,), "rank-out first value"
            )
            self._rankout_bindings[use_value_mix] = binding
        kernel(*ranks, *weights, *vectors, value_base, first_value, *outputs)
        return outputs

    def rankout_reduced(
        self,
        ranks: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        vectors: tuple[torch.Tensor, torch.Tensor],
        value_base: torch.Tensor,
        first_value: torch.Tensor,
        workspace: TileLangDecodeWorkspace,
        *,
        use_value_mix: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run rank-parallel W/A/G/V output from transposed weights."""
        outputs = (
            workspace.rank_decay,
            workspace.rank_gate_a,
            workspace.rank_gate_g,
            workspace.rank_value,
        )
        tensors = (*ranks, *weights, *vectors, value_base, first_value, *outputs)
        self._reject_training(tensors)
        kernel = self._rankout_reduced_kernels.get(use_value_mix)
        if kernel is None:
            kernel = _compiled_rankout_reduced(
                self.spec.channels,
                *self.spec.ranks,
                use_value_mix,
                self.input_dtype,
                4,
                128,
                self._architecture,
            )
            self._rankout_reduced_kernels[use_value_mix] = kernel
        binding = tuple(tensor.data_ptr() for tensor in tensors)
        if self._rankout_reduced_bindings.get(use_value_mix) != binding:
            for rank, size in zip(ranks, self.spec.ranks, strict=True):
                self._validate(rank, (size,), "reduced rank-out input")
            for weight, size in zip(weights, self.spec.ranks, strict=True):
                self._validate(
                    weight,
                    (self.spec.channels, size),
                    "transposed reduced rank-out weight",
                )
            for vector in vectors:
                self._validate(
                    vector, (self.spec.channels,), "reduced rank-out vector"
                )
            self._validate(
                value_base, (self.spec.channels,), "reduced rank-out value"
            )
            self._validate(
                first_value, (self.spec.channels,), "reduced rank-out first value"
            )
            self._rankout_reduced_bindings[use_value_mix] = binding
        kernel(*ranks, *weights, *vectors, value_base, first_value, *outputs)
        return outputs


    def key_gate(
        self,
        key: torch.Tensor,
        key_scale: torch.Tensor,
        gate_a: torch.Tensor,
        gate_scale: torch.Tensor,
        workspace: TileLangDecodeWorkspace,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fuse per-head key normalization and recurrent gate vectors."""
        outputs = (
            workspace.normalized_key,
            workspace.modified_key,
            workspace.anti_key,
            workspace.anti_gate,
        )
        tensors = (key, key_scale, gate_a, gate_scale, *outputs)
        self._reject_training(tensors)
        if self._key_gate_kernel is None:
            self._key_gate_kernel = _compiled_key_gate(
                self.spec.num_heads,
                self.spec.head_size,
                self.input_dtype,
                self._architecture,
            )
        for tensor in tensors:
            self._validate(
                tensor, (self.spec.channels,), "key normalization/gate tensor"
            )
        self._key_gate_kernel(*tensors)
        return outputs

    def tmix_layernorm_mix6(
        self,
        residual: torch.Tensor,
        previous: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        mix_weights: torch.Tensor,
        epsilon: float,
        workspace: TileLangDecodeWorkspace,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse LayerNorm and six shifted time-mix vectors."""
        normalized = previous
        mixed = workspace.tmix_mixed
        tensors = (
            residual,
            previous,
            norm_weight,
            norm_bias,
            mix_weights,
            normalized,
            mixed,
        )
        self._reject_training(tensors)
        if self._tmix_layernorm_mix6_kernel is None:
            self._tmix_layernorm_mix6_kernel = _compiled_tmix_layernorm_mix6(
                self.spec.channels,
                self.input_dtype,
                epsilon,
                256,
                self._architecture,
            )
        for tensor in tensors[:-1]:
            expected = (
                (6, self.spec.channels)
                if tensor is mix_weights
                else (self.spec.channels,)
            )
            self._validate(tensor, expected, "time-mix LayerNorm tensor")
        self._validate(
            mixed, (6, self.spec.channels), "time-mix mixed output"
        )
        self._tmix_layernorm_mix6_kernel(*tensors)
        return normalized, mixed

    def post_state(
        self,
        projected: torch.Tensor,
        receptance: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        r_k: torch.Tensor,
        gate: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        epsilon: float,
        workspace: TileLangDecodeWorkspace,
    ) -> torch.Tensor:
        """Fuse GroupNorm, RKV residual, and gate finalization."""
        output = workspace.post_mixed
        tensors = (
            projected,
            receptance,
            key,
            value,
            r_k,
            gate,
            norm_weight,
            norm_bias,
            output,
        )
        self._reject_training(tensors)
        if self._post_state_kernel is None:
            self._post_state_kernel = _compiled_post_state(
                self.spec.num_heads,
                self.spec.head_size,
                self.input_dtype,
                epsilon,
                self._architecture,
            )
        binding = tuple(tensor.data_ptr() for tensor in tensors)
        if self._post_state_binding != binding:
            head_shape = (self.spec.num_heads, self.spec.head_size)
            for tensor in (projected, receptance, key, value, r_k):
                self._validate(tensor, head_shape, "post-state head tensor")
            for tensor in (gate, norm_weight, norm_bias, output):
                self._validate(
                    tensor, (self.spec.channels,), "post-state channel tensor"
                )
            self._post_state_binding = binding
        self._post_state_kernel(*tensors)
        return output


    def cmix_layernorm_mix(
        self,
        residual: torch.Tensor,
        previous: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        mix_weight: torch.Tensor,
        epsilon: float,
        workspace: TileLangDecodeWorkspace,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse LayerNorm and shifted channel-mix input generation."""
        normalized = workspace.ffn_normalized
        mixed = workspace.post_mixed
        tensors = (
            residual,
            previous,
            norm_weight,
            norm_bias,
            mix_weight,
            normalized,
            mixed,
        )
        self._reject_training(tensors)
        if self._cmix_layernorm_mix_kernel is None:
            self._cmix_layernorm_mix_kernel = _compiled_cmix_layernorm_mix(
                self.spec.channels,
                self.input_dtype,
                epsilon,
                256,
                self._architecture,
            )
        for tensor in tensors:
            self._validate(
                tensor, (self.spec.channels,), "channel-mix LayerNorm tensor"
            )
        self._cmix_layernorm_mix_kernel(*tensors)
        return normalized, mixed

    def cmix_add_layernorm_mix(
        self,
        residual: torch.Tensor,
        update: torch.Tensor,
        previous: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        mix_weight: torch.Tensor,
        epsilon: float,
        workspace: TileLangDecodeWorkspace,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fuse residual add, LayerNorm, and shifted channel-mix input."""
        combined = workspace.block_residual
        normalized = previous
        mixed = workspace.post_mixed
        tensors = (
            residual,
            update,
            previous,
            norm_weight,
            norm_bias,
            mix_weight,
            combined,
            normalized,
            mixed,
        )
        self._reject_training(tensors)
        if self._cmix_add_layernorm_mix_kernel is None:
            self._cmix_add_layernorm_mix_kernel = (
                _compiled_cmix_add_layernorm_mix(
                    self.spec.channels,
                    self.input_dtype,
                    epsilon,
                    256,
                    self._architecture,
                )
            )
        for tensor in tensors:
            self._validate(
                tensor, (self.spec.channels,), "channel-mix add/LayerNorm tensor"
            )
        self._cmix_add_layernorm_mix_kernel(*tensors)
        return combined, normalized, mixed

    def ffn(
        self,
        mixed: torch.Tensor,
        key_weight: torch.Tensor,
        packed_value_weight: torch.Tensor,
        output: torch.Tensor,
        workspace: TileLangDecodeWorkspace,
        residual: torch.Tensor | None = None,
    ) -> None:
        """Run tiled key GEMV plus sparse TileLang down projection."""
        tensors = (mixed, key_weight, packed_value_weight, output)
        if residual is not None:
            tensors = (*tensors, residual)
        self._reject_training(tensors)
        self._validate(mixed, (self.spec.channels,), "FFN mixed")
        self._validate(
            key_weight,
            (self.spec.ffn_rows, self.spec.channels),
            "FFN key weight",
        )
        expected_value_shape = (self.spec.ffn_rows, self.spec.channels)
        self._validate(
            packed_value_weight,
            expected_value_shape,
            "FFN value weight",
        )
        self._validate(output, (self.spec.channels,), "FFN output")
        if residual is not None:
            self._validate(residual, (self.spec.channels,), "FFN residual")

        if self.dtype == torch.float16:
            if self._cmix_sparse_binned_kernel is None:
                self._cmix_sparse_binned_kernel = _compiled_cmix_sparse_binned(
                    self.spec.channels,
                    self.spec.ffn_rows,
                    self.input_dtype,
                    128,
                    128,
                    6,
                    self._architecture,
                )
                self._cmix_binned_finalize_kernel = (
                    _compiled_cmix_binned_finalize(
                        self.spec.channels,
                        6,
                        self.input_dtype,
                        256,
                        self._architecture,
                    )
                )
            self.gemv(mixed, key_weight, workspace.ffn_hidden)
            if self._cmix_binned_finalize_kernel is None:
                raise RuntimeError("binned FFN finalizer failed to initialize")
            self._cmix_sparse_binned_kernel(
                workspace.ffn_hidden,
                packed_value_weight,
                workspace.ffn_bins,
            )
            residual_input = output if residual is None else residual
            self._cmix_binned_finalize_kernel(
                workspace.ffn_bins, residual_input, output
            )
            return

        split_rows = self.spec.ffn_rows // 4
        if self.spec.ffn_rows % 4:
            raise ValueError("FFN rows must be divisible by four streams")
        if self._ffn_kernel is None:
            self._ffn_kernel = _compiled_ffn(
                self.spec.channels,
                split_rows,
                2,
                128,
                self.input_dtype,
                self._architecture,
            )
            self._cmix_value_kernel = _compiled_cmix_value(
                self.spec.channels,
                split_rows,
                self.input_dtype,
                64,
                8,
                self._architecture,
            )
            self._cmix_finalize_kernel = _compiled_cmix_finalize(
                self.spec.channels,
                4,
                self.input_dtype,
                self._architecture,
            )
        if (
            self._ffn_kernel is None
            or self._cmix_value_kernel is None
            or self._cmix_finalize_kernel is None
        ):
            raise RuntimeError("TileLang BF16 FFN kernels failed to initialize")
        current_stream = torch.cuda.current_stream(self.device)
        for split, stream in enumerate(self._streams):
            start = split * split_rows
            stop = start + split_rows
            stream.wait_stream(current_stream)
            with torch.cuda.stream(stream):
                self._ffn_kernel(
                    mixed,
                    key_weight[start:stop],
                    workspace.ffn_hidden[start:stop],
                )
                self._cmix_value_kernel(
                    workspace.ffn_hidden[start:stop],
                    packed_value_weight[start:stop],
                    workspace.ffn_partials[split],
                )
        for stream in self._streams:
            current_stream.wait_stream(stream)
        self._cmix_finalize_kernel(workspace.ffn_partials, output)

    def wkv_w0_t1(
        self,
        state: torch.Tensor,
        receptance: torch.Tensor,
        decay: torch.Tensor,
        decay_bias: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate_a: torch.Tensor,
        gate_b: torch.Tensor,
        elapsed: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Run specialized B1T1 FP16 WKV with fused static decay bias."""
        self._reject_training(
            (
                state,
                receptance,
                decay,
                decay_bias,
                key,
                value,
                gate_a,
                gate_b,
                output,
            )
        )
        _wkv_w0_t1_out(
            state,
            receptance,
            decay,
            decay_bias,
            key,
            value,
            gate_a,
            gate_b,
            elapsed,
            output,
        )

    def wkv(
        self,
        state: torch.Tensor,
        receptance: torch.Tensor,
        decay_raw: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate_a: torch.Tensor,
        gate_b: torch.Tensor,
        elapsed: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Run exact FP16 recurrent update into caller-owned buffers."""
        self._reject_training(
            (state, receptance, decay_raw, key, value, gate_a, gate_b, output)
        )
        _wkv_out(
            state,
            receptance,
            decay_raw,
            key,
            value,
            gate_a,
            gate_b,
            elapsed,
            output,
        )

    def _validate(
        self, tensor: torch.Tensor, shape: tuple[int, ...], name: str
) -> None:
        if tensor.device != self.device:
            raise ValueError(f"{name} must be on {self.device}")
        if tensor.dtype != self.dtype:
            raise TypeError(f"{name} must use {self.dtype}")
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    @staticmethod
    def _reject_training(tensors: Sequence[torch.Tensor]) -> None:
        if torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in tensors
        ):
            raise RuntimeError("TileLang decode backend is inference-only")


def clear_tilelang_runtime_caches() -> None:
    """Synchronize and release the bounded per-device auxiliary stream pools."""
    for device_index in tuple(_STREAM_POOLS):
        with torch.cuda.device(device_index):
            torch.cuda.synchronize(device_index)
    _STREAM_POOLS.clear()
