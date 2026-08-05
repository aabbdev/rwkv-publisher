# pyright: reportInvalidTypeForm=false
from functools import lru_cache
from typing import Any

import torch

FLOAT16 = torch.float16  # type: ignore[attr-defined]
BFLOAT16 = torch.bfloat16  # type: ignore[attr-defined]
FLOAT32 = torch.float32  # type: ignore[attr-defined]
TORCH_WHERE = torch.where  # type: ignore[attr-defined]
TORCH_ZEROS_LIKE = torch.zeros_like  # type: ignore[attr-defined]
TORCH_STACK = torch.stack  # type: ignore[attr-defined]
CUDA_GET_DEVICE_CAPABILITY = torch.cuda.get_device_capability
IS_GRAD_ENABLED = torch.is_grad_enabled  # type: ignore[attr-defined]


def cuda_arch_key(device: Any | None = None) -> str:
    major, minor = CUDA_GET_DEVICE_CAPABILITY(device)
    return f"sm_{major}{minor}"


EXACT_FUSED_STATE_PROJECTION_CAPABILITIES = frozenset({(8, 9)})


def exact_fused_state_projection_supported(device: Any | None = None) -> bool:
    """Return whether padded narrow projection is bit-exact on this GPU."""
    return (
        tuple(CUDA_GET_DEVICE_CAPABILITY(device))
        in EXACT_FUSED_STATE_PROJECTION_CAPABILITIES
    )


def _require_exact_fused_state_projection(device: Any | None = None) -> None:
    if not exact_fused_state_projection_supported(device):
        capability = CUDA_GET_DEVICE_CAPABILITY(device)
        raise RuntimeError(
            "Exact fused state projection is not validated for CUDA capability "
            f"{capability}; use tilelang_state_update for exact fallback"
        )


def build_state_program(
    batch_size: int,
    num_heads: int,
    head_size: int,
    input_dtype: str,
):
    """Build exact pointwise RWKV state finalization as a TileLang PrimFunc."""
    import tilelang.language as T  # type: ignore[import-not-found]

    @T.prim_func
    def kernel(
        state: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, num_heads, head_size, head_size), "float32"
        ),
        decay: T.Tensor((batch_size, num_heads, head_size), input_dtype),  # type: ignore[reportInvalidTypeForm]
        anti_update: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, num_heads, head_size, head_size), "float32"
        ),
        value_key: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, num_heads, head_size, head_size), "float32"
        ),
        next_state: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, num_heads, head_size, head_size), "float32"
        ),
    ):
        with T.Kernel(batch_size, num_heads, threads=256) as (batch, head):
            # Preserve eager PyTorch's FP32 rounding boundaries exactly:
            # multiply -> store, add anti update -> store, add value/key -> store.
            for row, column in T.Parallel(head_size, head_size):
                next_state[batch, head, row, column] = state[
                    batch, head, row, column
                ] * T.cast(decay[batch, head, column], "float32")
            T.sync_threads()
            for row, column in T.Parallel(head_size, head_size):
                next_state[batch, head, row, column] = (
                    next_state[batch, head, row, column]
                    + anti_update[batch, head, row, column]
                )
            T.sync_threads()
            for row, column in T.Parallel(head_size, head_size):
                next_state[batch, head, row, column] = (
                    next_state[batch, head, row, column]
                    + value_key[batch, head, row, column]
                )

    return kernel


def build_state_backward_program(
    batch_size: int,
    num_heads: int,
    head_size: int,
    input_dtype: str,
):
    """Build first-order gradients for exact pointwise state finalization."""
    import tilelang.language as T  # type: ignore[import-not-found]

    @T.prim_func
    def kernel(
        grad_output: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, num_heads, head_size, head_size), "float32"
        ),
        decay: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, num_heads, head_size), input_dtype
        ),
        grad_state: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, num_heads, head_size, head_size), "float32"
        ),
    ):
        with T.Kernel(batch_size, num_heads, threads=256) as (batch, head):
            for row, column in T.Parallel(head_size, head_size):
                grad_state[batch, head, row, column] = grad_output[
                    batch, head, row, column
                ] * T.cast(decay[batch, head, column], "float32")

    return kernel


def build_low_precision_state_program(
    batch_size: int,
    num_heads: int,
    head_size: int,
    input_dtype: str,
):
    """Build FP32-compute finalization with FP16 or BF16 storage."""
    import tilelang.language as T  # type: ignore[import-not-found]

    @T.prim_func
    def kernel(
        state: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, num_heads, head_size, head_size), input_dtype
        ),
        decay: T.Tensor((batch_size, num_heads, head_size), input_dtype),  # type: ignore[reportInvalidTypeForm]
        anti_update: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, num_heads, head_size, head_size), "float32"
        ),
        value_key: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, num_heads, head_size, head_size), "float32"
        ),
        next_state: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, num_heads, head_size, head_size), input_dtype
        ),
    ):
        with T.Kernel(batch_size, num_heads, threads=256) as (batch, head):
            workspace = T.alloc_shared((head_size, head_size), "float32")
            for row, column in T.Parallel(head_size, head_size):
                workspace[row, column] = T.cast(
                    state[batch, head, row, column], "float32"
                ) * T.cast(decay[batch, head, column], "float32")
            T.sync_threads()
            for row, column in T.Parallel(head_size, head_size):
                workspace[row, column] = (
                    workspace[row, column]
                    + anti_update[batch, head, row, column]
                )
            T.sync_threads()
            for row, column in T.Parallel(head_size, head_size):
                workspace[row, column] = (
                    workspace[row, column]
                    + value_key[batch, head, row, column]
                )
            T.sync_threads()
            for row, column in T.Parallel(head_size, head_size):
                next_state[batch, head, row, column] = T.cast(
                    workspace[row, column], input_dtype
                )

    return kernel


def build_state_projection_program(
    batch_size: int,
    num_heads: int,
    head_size: int,
    input_dtype: str,
):
    """Build padded Tensor Core state-by-receptance projection."""
    import tilelang.language as T  # type: ignore[import-not-found]

    padded_columns = 16

    @T.prim_func
    def kernel(
        state: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, num_heads, head_size, head_size), input_dtype
        ),
        receptance: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, num_heads, head_size), input_dtype
        ),
        mixed: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, num_heads, head_size), input_dtype
        ),
    ):
        with T.Kernel(batch_size, num_heads, threads=128) as (batch, head):
            state_shared = T.alloc_shared(
                (head_size, head_size), input_dtype
            )
            receptance_shared = T.alloc_shared(
                (head_size, padded_columns), input_dtype
            )
            mixed_fragment = T.alloc_fragment(
                (head_size, padded_columns), "float32"
            )
            mixed_shared = T.alloc_shared(
                (head_size, padded_columns), input_dtype
            )
            T.copy(state[batch, head, :, :], state_shared)
            for row, column in T.Parallel(head_size, padded_columns):
                receptance_shared[row, column] = receptance[batch, head, row]
            T.clear(mixed_fragment)
            T.gemm(state_shared, receptance_shared, mixed_fragment)
            T.copy(mixed_fragment, mixed_shared)
            T.sync_threads()
            for row in T.Parallel(head_size):
                mixed[batch, head, row] = mixed_shared[row, 0]

    return kernel


def build_fused_state_update_program(
    batch_size: int,
    num_heads: int,
    head_size: int,
    input_dtype: str,
):
    """Build exact finalization plus padded Tensor Core projection."""
    import tilelang.language as T  # type: ignore[import-not-found]

    padded_columns = 16

    @T.prim_func
    def kernel(
        state: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, num_heads, head_size, head_size), "float32"
        ),
        decay: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, num_heads, head_size), input_dtype
        ),
        anti_update: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, num_heads, head_size, head_size), "float32"
        ),
        value_key: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, num_heads, head_size, head_size), "float32"
        ),
        receptance: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, num_heads, head_size), input_dtype
        ),
        next_state: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, num_heads, head_size, head_size), "float32"
        ),
        mixed: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, num_heads, head_size), input_dtype
        ),
    ):
        with T.Kernel(batch_size, num_heads, threads=256) as (batch, head):
            state_shared = T.alloc_shared(
                (head_size, head_size), input_dtype
            )
            receptance_shared = T.alloc_shared(
                (head_size, padded_columns), input_dtype
            )
            mixed_fragment = T.alloc_fragment(
                (head_size, padded_columns), "float32"
            )
            mixed_shared = T.alloc_shared(
                (head_size, padded_columns), input_dtype
            )

            # Preserve eager FP32 stores between multiply and both additions.
            for row, column in T.Parallel(head_size, head_size):
                next_state[batch, head, row, column] = state[
                    batch, head, row, column
                ] * T.cast(decay[batch, head, column], "float32")
            T.sync_threads()
            for row, column in T.Parallel(head_size, head_size):
                next_state[batch, head, row, column] = (
                    next_state[batch, head, row, column]
                    + anti_update[batch, head, row, column]
                )
            T.sync_threads()
            for row, column in T.Parallel(head_size, head_size):
                next_state[batch, head, row, column] = (
                    next_state[batch, head, row, column]
                    + value_key[batch, head, row, column]
                )
            T.sync_threads()

            for row, column in T.Parallel(head_size, head_size):
                state_shared[row, column] = T.cast(
                    next_state[batch, head, row, column], input_dtype
                )
            for row, column in T.Parallel(head_size, padded_columns):
                receptance_shared[row, column] = receptance[batch, head, row]
            T.clear(mixed_fragment)
            T.gemm(state_shared, receptance_shared, mixed_fragment)
            T.copy(mixed_fragment, mixed_shared)
            T.sync_threads()
            for row in T.Parallel(head_size):
                mixed[batch, head, row] = mixed_shared[row, 0]

    return kernel








def build_x_mix_program(
    batch_size: int,
    hidden_size: int,
    input_dtype: str,
):
    """Build exact six-way decode x-mix without temporary stacking."""
    import tilelang.language as T  # type: ignore[import-not-found]

    block_size = 256

    @T.prim_func
    def kernel(
        x: T.Tensor((batch_size, hidden_size), input_dtype),  # type: ignore[reportInvalidTypeForm]
        previous: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, hidden_size), input_dtype
        ),
        mix_r: T.Tensor((hidden_size,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        mix_w: T.Tensor((hidden_size,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        mix_k: T.Tensor((hidden_size,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        mix_v: T.Tensor((hidden_size,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        mix_a: T.Tensor((hidden_size,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        mix_g: T.Tensor((hidden_size,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        mixed: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, 6, hidden_size), input_dtype
        ),
    ):
        with T.Kernel(
            T.ceildiv(hidden_size, block_size), batch_size, threads=block_size
        ) as (block, batch):
            delta_shared = T.alloc_shared((block_size,), input_dtype)
            products_shared = T.alloc_shared((6, block_size), input_dtype)

            for lane in T.Parallel(block_size):
                column = block * block_size + lane
                if column < hidden_size:
                    delta_shared[lane] = T.cast(
                        previous[batch, column] - x[batch, column], input_dtype
                    )
            T.sync_threads()

            for lane in T.Parallel(block_size):
                column = block * block_size + lane
                if column < hidden_size:
                    products_shared[0, lane] = T.cast(
                        delta_shared[lane] * mix_r[column], input_dtype
                    )
                    products_shared[1, lane] = T.cast(
                        delta_shared[lane] * mix_w[column], input_dtype
                    )
                    products_shared[2, lane] = T.cast(
                        delta_shared[lane] * mix_k[column], input_dtype
                    )
                    products_shared[3, lane] = T.cast(
                        delta_shared[lane] * mix_v[column], input_dtype
                    )
                    products_shared[4, lane] = T.cast(
                        delta_shared[lane] * mix_a[column], input_dtype
                    )
                    products_shared[5, lane] = T.cast(
                        delta_shared[lane] * mix_g[column], input_dtype
                    )
            T.sync_threads()

            for lane in T.Parallel(block_size):
                column = block * block_size + lane
                if column < hidden_size:
                    for mix_index in T.Serial(6):
                        mixed[batch, mix_index, column] = T.cast(
                            x[batch, column]
                            + products_shared[mix_index, lane],
                            input_dtype,
                        )

    return kernel


def build_post_state_program(
    batch_size: int,
    num_heads: int,
    head_size: int,
    input_dtype: str,
):
    """Build exact fused RKV correction and gating after native GroupNorm."""
    import tilelang.language as T  # type: ignore[import-not-found]

    @T.prim_func
    def kernel(
        normalized: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, num_heads, head_size), input_dtype
        ),
        receptance: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, num_heads, head_size), input_dtype
        ),
        key: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, num_heads, head_size), input_dtype
        ),
        value: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, num_heads, head_size), input_dtype
        ),
        r_k: T.Tensor((num_heads, head_size), input_dtype),  # type: ignore[reportInvalidTypeForm]
        gate: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, num_heads, head_size), input_dtype
        ),
        output: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, num_heads, head_size), input_dtype
        ),
    ):
        with T.Kernel(batch_size, num_heads, threads=32) as (batch, head):
            product_stage = T.alloc_shared((head_size,), input_dtype)
            correction = T.alloc_shared((head_size,), input_dtype)
            corrected = T.alloc_shared((head_size,), input_dtype)
            rkv_sum = T.alloc_shared((1,), input_dtype)

            for column in T.Parallel(head_size):
                product_stage[column] = T.cast(
                    receptance[batch, head, column]
                    * key[batch, head, column],
                    input_dtype,
                )
            T.sync_threads()

            for column in T.Parallel(head_size):
                product_stage[column] = T.cast(
                    product_stage[column] * r_k[head, column], input_dtype
                )
            T.sync_threads()

            for worker in T.Parallel(1):
                total = T.alloc_local((1,), "float32")
                total[0] = 0.0
                for column in T.serial(head_size):
                    total[0] += T.cast(product_stage[column], "float32")
                rkv_sum[0] = T.cast(total[0], input_dtype)
            T.sync_threads()

            for column in T.Parallel(head_size):
                correction[column] = T.cast(
                    rkv_sum[0] * value[batch, head, column], input_dtype
                )
            T.sync_threads()

            for column in T.Parallel(head_size):
                corrected[column] = T.cast(
                    normalized[batch, head, column] + correction[column],
                    input_dtype,
                )
            T.sync_threads()

            for column in T.Parallel(head_size):
                output[batch, head, column] = T.cast(
                    corrected[column] * gate[batch, head, column], input_dtype
                )

    return kernel


def _dtype_name(dtype) -> str:
    if dtype == FLOAT16:
        return "float16"
    if dtype == BFLOAT16:
        return "bfloat16"
    if dtype == FLOAT32:
        return "float32"
    raise TypeError(f"TileLang RWKV kernel does not support dtype {dtype}")


@lru_cache(maxsize=32)
def _compiled_kernel(
    batch_size: int,
    num_heads: int,
    head_size: int,
    input_dtype: str,
    device_arch: str,
):
    del device_arch
    import tilelang  # type: ignore[import-not-found]

    program = build_state_program(
        batch_size,
        num_heads,
        head_size,
        input_dtype,
    )
    return tilelang.compile(
        program,
        out_idx=-1,
        execution_backend="auto",
    )


@lru_cache(maxsize=32)
def _compiled_backward_kernel(
    batch_size: int,
    num_heads: int,
    head_size: int,
    input_dtype: str,
    device_arch: str,
):
    del device_arch
    import tilelang  # type: ignore[import-not-found]

    return tilelang.compile(
        build_state_backward_program(
            batch_size, num_heads, head_size, input_dtype
        ),
        out_idx=-1,
        execution_backend="auto",
    )

@lru_cache(maxsize=32)
def _compiled_low_precision_state_kernel(
    batch_size: int,
    num_heads: int,
    head_size: int,
    input_dtype: str,
    device_arch: str,
):
    del device_arch
    import tilelang  # type: ignore[import-not-found]

    return tilelang.compile(
        build_low_precision_state_program(
            batch_size, num_heads, head_size, input_dtype
        ),
        out_idx=-1,
        execution_backend="auto",
    )


def torch_state_update(
    state: torch.Tensor,
    decay: torch.Tensor,
    normalized_key: torch.Tensor,
    gate_a: torch.Tensor,
    value: torch.Tensor,
    key: torch.Tensor,
    receptance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    anti_matrix = (-normalized_key).unsqueeze(-1) @ (normalized_key * gate_a).unsqueeze(
        -2
    )
    value_key = value.unsqueeze(-1) @ key.unsqueeze(-2)
    state_f32 = state.float()
    next_state = state_f32 * decay.float().unsqueeze(-2)
    next_state = next_state + state_f32 @ anti_matrix.float()
    next_state = next_state + value_key.float()
    if state.dtype == BFLOAT16:
        next_state = next_state.to(BFLOAT16)
    mixed = (next_state.to(receptance.dtype) @ receptance.unsqueeze(-1)).squeeze(-1)
    return next_state, mixed


@lru_cache(maxsize=32)
def _compiled_x_mix(
    batch_size: int,
    hidden_size: int,
    input_dtype: str,
    device_arch: str,
):
    del device_arch
    import tilelang  # type: ignore[import-not-found]

    return tilelang.compile(
        build_x_mix_program(batch_size, hidden_size, input_dtype),
        out_idx=-1,
        execution_backend="auto",
    )


def tilelang_x_mix(
    x: torch.Tensor,
    previous: torch.Tensor,
    mix_r: torch.Tensor,
    mix_w: torch.Tensor,
    mix_k: torch.Tensor,
    mix_v: torch.Tensor,
    mix_a: torch.Tensor,
    mix_g: torch.Tensor,
) -> torch.Tensor:
    """Fuse exact six-way BF16/FP16 decode x-mix."""
    if x.device.type != "cuda" or any(
        tensor.device != x.device
        for tensor in (previous, mix_r, mix_w, mix_k, mix_v, mix_a, mix_g)
    ):
        raise RuntimeError("TileLang x-mix requires one CUDA device")
    if x.dtype != BFLOAT16 or any(
        tensor.dtype != x.dtype
        for tensor in (previous, mix_r, mix_w, mix_k, mix_v, mix_a, mix_g)
    ):
        raise TypeError("TileLang x-mix requires matching bfloat16 tensors")
    if x.ndim != 2 or previous.shape != x.shape:
        raise ValueError("x and previous must have shape [batch, hidden]")
    batch_size, hidden_size = x.shape
    if any(
        tensor.shape != (hidden_size,)
        for tensor in (mix_r, mix_w, mix_k, mix_v, mix_a, mix_g)
    ):
        raise ValueError("x-mix weights must have shape [hidden]")
    kernel: Any = _compiled_x_mix(
        batch_size,
        hidden_size,
        _dtype_name(x.dtype),
        cuda_arch_key(x.device),
    )
    return kernel(
        x.contiguous(),
        previous.contiguous(),
        mix_r.contiguous(),
        mix_w.contiguous(),
        mix_k.contiguous(),
        mix_v.contiguous(),
        mix_a.contiguous(),
        mix_g.contiguous(),
    )


@lru_cache(maxsize=32)
def _compiled_post_state(
    batch_size: int,
    num_heads: int,
    head_size: int,
    input_dtype: str,
    device_arch: str,
):
    del device_arch
    import tilelang  # type: ignore[import-not-found]

    return tilelang.compile(
        build_post_state_program(batch_size, num_heads, head_size, input_dtype),
        out_idx=-1,
        execution_backend="auto",
    )


def tilelang_post_state(
    normalized: torch.Tensor,
    receptance: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    r_k: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Fuse exact BF16 RKV correction and gating after native GroupNorm."""
    inputs = (receptance, key, value, gate)
    if normalized.device.type != "cuda" or any(
        tensor.device != normalized.device for tensor in (*inputs, r_k)
    ):
        raise RuntimeError("TileLang post-state fusion requires one CUDA device")
    if normalized.dtype != BFLOAT16 or any(
        tensor.dtype != normalized.dtype for tensor in (*inputs, r_k)
    ):
        raise TypeError("TileLang post-state fusion requires matching bfloat16")
    if normalized.ndim != 3:
        raise ValueError("normalized must have shape [batch, heads, head]")
    batch_size, num_heads, head_size = normalized.shape
    expected = (batch_size, num_heads, head_size)
    if any(tensor.shape != expected for tensor in inputs):
        raise ValueError("post-state vector shapes are incompatible")
    if r_k.shape != (num_heads, head_size):
        raise ValueError("r_k must have shape [heads, head]")
    kernel: Any = _compiled_post_state(
        batch_size,
        num_heads,
        head_size,
        _dtype_name(normalized.dtype),
        cuda_arch_key(normalized.device),
    )
    return kernel(
        normalized.contiguous(),
        receptance.contiguous(),
        key.contiguous(),
        value.contiguous(),
        r_k.contiguous(),
        gate.contiguous(),
    )


@lru_cache(maxsize=32)
def _compiled_fused_state_update(
    batch_size: int,
    num_heads: int,
    head_size: int,
    input_dtype: str,
    device_arch: str,
):
    del device_arch
    import tilelang  # type: ignore[import-not-found]

    return tilelang.compile(
        build_fused_state_update_program(
            batch_size, num_heads, head_size, input_dtype
        ),
        out_idx=[-2, -1],
        execution_backend="auto",
    )




def tilelang_fused_state_update(
    state: torch.Tensor,
    decay: torch.Tensor,
    anti_update: torch.Tensor,
    value_key: torch.Tensor,
    receptance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse exact FP32 state finalization and validated narrow projection."""
    batch_size, num_heads, head_size, columns = state.shape
    if columns != head_size or head_size % 16:
        raise ValueError("fused state update requires square 16-aligned heads")
    if state.device.type != "cuda" or any(
        tensor.device != state.device
        for tensor in (decay, anti_update, value_key, receptance)
    ):
        raise RuntimeError("fused state update requires one CUDA device")
    _require_exact_fused_state_projection(state.device)
    if state.dtype != FLOAT32 or anti_update.dtype != FLOAT32 or value_key.dtype != FLOAT32:
        raise TypeError("fused state update requires FP32 state updates")
    if decay.dtype != receptance.dtype or decay.dtype not in {FLOAT16, BFLOAT16}:
        raise TypeError("fused state update requires matching float16/bfloat16 vectors")
    kernel: Any = _compiled_fused_state_update(
        batch_size,
        num_heads,
        head_size,
        _dtype_name(decay.dtype),
        cuda_arch_key(state.device),
    )
    return kernel(
        state.contiguous(),
        decay.contiguous(),
        anti_update.contiguous(),
        value_key.contiguous(),
        receptance.contiguous(),
    )








@lru_cache(maxsize=32)
def _compiled_state_projection(
    batch_size: int,
    num_heads: int,
    head_size: int,
    input_dtype: str,
    device_arch: str,
):
    del device_arch
    import tilelang  # type: ignore[import-not-found]

    return tilelang.compile(
        build_state_projection_program(
            batch_size, num_heads, head_size, input_dtype
        ),
        out_idx=-1,
        execution_backend="auto",
    )


def tilelang_state_projection(
    state: torch.Tensor, receptance: torch.Tensor
) -> torch.Tensor:
    """Project head state on GPUs with validated exact padded MMA behavior."""
    if state.device.type != "cuda" or receptance.device != state.device:
        raise RuntimeError("TileLang state projection requires one CUDA device")
    _require_exact_fused_state_projection(state.device)
    if state.dtype != receptance.dtype or state.dtype not in {FLOAT16, BFLOAT16}:
        raise TypeError("TileLang state projection requires matching float16/bfloat16")
    if state.ndim != 4 or receptance.ndim != 3:
        raise ValueError("state/receptance ranks must be 4 and 3")
    batch_size, num_heads, head_size, columns = state.shape
    if columns != head_size or receptance.shape != (batch_size, num_heads, head_size):
        raise ValueError("state/receptance shapes are incompatible")
    if head_size % 16:
        raise ValueError("TileLang state projection requires head size divisible by 16")
    kernel: Any = _compiled_state_projection(
        batch_size,
        num_heads,
        head_size,
        _dtype_name(state.dtype),
        cuda_arch_key(state.device),
    )
    return kernel(state.contiguous(), receptance.contiguous())


def tilelang_state_finalize(
    state: torch.Tensor,
    decay: torch.Tensor,
    anti_update: torch.Tensor,
    value_key: torch.Tensor,
) -> torch.Tensor:
    """Run compiled pointwise state finalization without PyTorch dispatch."""
    batch_size, num_heads, head_size, _ = state.shape
    kernel: Any = _compiled_kernel(
        batch_size,
        num_heads,
        head_size,
        _dtype_name(decay.dtype),
        cuda_arch_key(state.device),
    )
    return kernel(
        state.contiguous(),
        decay.contiguous(),
        anti_update.contiguous(),
        value_key.contiguous(),
    )


def tilelang_state_finalize_backward(
    grad_output: torch.Tensor, decay: torch.Tensor
) -> torch.Tensor:
    """Run the compiled first-order state-gradient kernel."""
    batch_size, num_heads, head_size, _ = grad_output.shape
    kernel: Any = _compiled_backward_kernel(
        batch_size,
        num_heads,
        head_size,
        _dtype_name(decay.dtype),
        cuda_arch_key(grad_output.device),
    )
    return kernel(grad_output.contiguous(), decay.contiguous())


def tilelang_low_precision_state_finalize(
    state: torch.Tensor,
    decay: torch.Tensor,
    anti_update: torch.Tensor,
    value_key: torch.Tensor,
) -> torch.Tensor:
    """Finalize recurrent state with FP32 compute and low-precision storage."""
    if state.dtype not in {FLOAT16, BFLOAT16} or decay.dtype != state.dtype:
        raise TypeError(
            "Low-precision state finalization requires matching FP16/BF16 state and decay"
        )
    if anti_update.dtype != FLOAT32 or value_key.dtype != FLOAT32:
        raise TypeError("Low-precision state finalization requires FP32 updates")
    batch_size, num_heads, head_size, columns = state.shape
    if columns != head_size:
        raise ValueError("Low-precision recurrent state must be square per head")
    kernel: Any = _compiled_low_precision_state_kernel(
        batch_size,
        num_heads,
        head_size,
        _dtype_name(decay.dtype),
        cuda_arch_key(state.device),
    )
    return kernel(
        state.contiguous(),
        decay.contiguous(),
        anti_update.contiguous(),
        value_key.contiguous(),
    )




def tilelang_state_update(
    state: torch.Tensor,
    decay: torch.Tensor,
    normalized_key: torch.Tensor,
    gate_a: torch.Tensor,
    value: torch.Tensor,
    key: torch.Tensor,
    receptance: torch.Tensor,
    *,
    state_finalize_op: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if state.device.type != "cuda":
        raise RuntimeError("TileLang RWKV kernel requires CUDA tensors")

    state = state.contiguous()
    decay = decay.contiguous()
    normalized_key = normalized_key.contiguous()
    gate_a = gate_a.contiguous()
    value = value.contiguous()
    key = key.contiguous()
    receptance = receptance.contiguous()

    # Keep BF16/FP16 outer products and FP32 batched GEMM in PyTorch. Their
    # accumulation order is observable after long recurrent sequences.
    anti_matrix = (-normalized_key).unsqueeze(-1) @ (normalized_key * gate_a).unsqueeze(
        -2
    )
    state_f32 = state.float()
    anti_update = state_f32 @ anti_matrix.float()
    value_key = (value.unsqueeze(-1) @ key.unsqueeze(-2)).float()
    if state.dtype in {FLOAT16, BFLOAT16}:
        next_state = tilelang_low_precision_state_finalize(
            state, decay, anti_update, value_key
        )
        mixed = (next_state @ receptance.unsqueeze(-1)).squeeze(-1)
        return next_state, mixed
    if state.dtype != FLOAT32:
        raise TypeError(
            "TileLang recurrent state must be float32, float16, or bfloat16"
        )

    differentiable = IS_GRAD_ENABLED() and any(
        tensor.requires_grad for tensor in (state, decay, anti_update, value_key)
    )
    if (
        not differentiable
        and receptance.dtype in {FLOAT16, BFLOAT16}
        and state.shape[-1] % 16 == 0
        and exact_fused_state_projection_supported(state.device)
    ):
        return tilelang_fused_state_update(
            state, decay, anti_update, value_key, receptance
        )
    if differentiable:
        if state_finalize_op is None:
            raise RuntimeError(
                "differentiable TileLang state update requires the registered custom op"
            )
        next_state = state_finalize_op(state, decay, anti_update, value_key)
    else:
        next_state = tilelang_state_finalize(state, decay, anti_update, value_key)
    mixed = (next_state.to(receptance.dtype) @ receptance.unsqueeze(-1)).squeeze(-1)
    return next_state, mixed


def torch_state_scan(
    state: torch.Tensor,
    decay: torch.Tensor,
    normalized_key: torch.Tensor,
    gate_a: torch.Tensor,
    value: torch.Tensor,
    key: torch.Tensor,
    receptance: torch.Tensor,
    *,
    active: torch.Tensor | None = None,
    reset: torch.Tensor | None = None,
    output_mode: str = "full",
    chunk_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact executable specification for recurrent sequence state updates."""
    if output_mode not in {"full", "final"}:
        raise ValueError("output_mode must be full or final")
    if chunk_size < 0:
        raise ValueError("chunk_size must be non-negative")
    if decay.ndim != 4:
        raise ValueError("Sequence tensors must have shape [batch, time, heads, head]")
    sequence_length = decay.shape[1]
    if sequence_length == 0:
        raise ValueError("State scan requires at least one token")
    step = chunk_size or sequence_length
    matrix = state
    mixed_steps: list[torch.Tensor] = []
    final_mixed: torch.Tensor | None = None
    for chunk_start in range(0, sequence_length, step):
        chunk_stop = min(chunk_start + step, sequence_length)
        for token_index in range(chunk_start, chunk_stop):
            if reset is not None:
                reset_token = reset[:, token_index].bool().reshape(-1, 1, 1, 1)
                matrix = TORCH_WHERE(reset_token, TORCH_ZEROS_LIKE(matrix), matrix)
            matrix_candidate, mixed_candidate = torch_state_update(
                matrix,
                decay[:, token_index],
                normalized_key[:, token_index],
                gate_a[:, token_index],
                value[:, token_index],
                key[:, token_index],
                receptance[:, token_index],
            )
            if active is None:
                matrix = matrix_candidate
                mixed = mixed_candidate
            else:
                active_token = active[:, token_index].bool().reshape(-1, 1, 1, 1)
                matrix = TORCH_WHERE(active_token, matrix_candidate, matrix)
                mixed = (
                    matrix.to(receptance.dtype)
                    @ receptance[:, token_index].unsqueeze(-1)
                ).squeeze(-1)
            if output_mode == "full":
                mixed_steps.append(mixed)
            final_mixed = mixed
    if output_mode == "full":
        return matrix, TORCH_STACK(mixed_steps, dim=1)
    if final_mixed is None:
        raise RuntimeError("State scan produced no output")
    return matrix, final_mixed.unsqueeze(1)


def tilelang_state_scan(
    state: torch.Tensor,
    decay: torch.Tensor,
    normalized_key: torch.Tensor,
    gate_a: torch.Tensor,
    value: torch.Tensor,
    key: torch.Tensor,
    receptance: torch.Tensor,
    *,
    active: torch.Tensor | None = None,
    reset: torch.Tensor | None = None,
    output_mode: str = "full",
    chunk_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact TileLang-assisted sequence scan using the validated state kernel."""
    if state.device.type != "cuda":
        raise RuntimeError("TileLang RWKV sequence scan requires CUDA tensors")
    if output_mode not in {"full", "final"}:
        raise ValueError("output_mode must be full or final")
    if chunk_size < 0:
        raise ValueError("chunk_size must be non-negative")
    sequence_length = decay.shape[1]
    if sequence_length == 0:
        raise ValueError("State scan requires at least one token")
    step = chunk_size or sequence_length
    matrix = state
    mixed_steps: list[torch.Tensor] = []
    final_mixed: torch.Tensor | None = None
    for chunk_start in range(0, sequence_length, step):
        chunk_stop = min(chunk_start + step, sequence_length)
        for token_index in range(chunk_start, chunk_stop):
            if reset is not None:
                reset_token = reset[:, token_index].bool().reshape(-1, 1, 1, 1)
                matrix = TORCH_WHERE(reset_token, TORCH_ZEROS_LIKE(matrix), matrix)
            candidate, candidate_mixed = tilelang_state_update(
                matrix,
                decay[:, token_index],
                normalized_key[:, token_index],
                gate_a[:, token_index],
                value[:, token_index],
                key[:, token_index],
                receptance[:, token_index],
            )
            if active is None:
                matrix = candidate
                mixed = candidate_mixed
            else:
                active_token = active[:, token_index].bool().reshape(-1, 1, 1, 1)
                matrix = TORCH_WHERE(active_token, candidate, matrix)
                mixed = (
                    matrix.to(receptance.dtype)
                    @ receptance[:, token_index].unsqueeze(-1)
                ).squeeze(-1)
            if output_mode == "full":
                mixed_steps.append(mixed)
            final_mixed = mixed
    if output_mode == "full":
        return matrix, TORCH_STACK(mixed_steps, dim=1)
    if final_mixed is None:
        raise RuntimeError("State scan produced no output")
    return matrix, final_mixed.unsqueeze(1)


def build_fast_state_scan_program(
    batch_size: int,
    sequence_length: int,
    num_heads: int,
    head_size: int,
    input_dtype: str,
):
    """Build experimental persistent recurrent scan with one block per head."""
    import tilelang.language as T  # type: ignore[import-not-found]

    @T.prim_func
    def kernel(
        state: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, num_heads, head_size, head_size), "float32"
        ),
        decay: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, sequence_length, num_heads, head_size), input_dtype
        ),
        normalized_key: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, sequence_length, num_heads, head_size), input_dtype
        ),
        gate_a: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, sequence_length, num_heads, head_size), input_dtype
        ),
        value: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, sequence_length, num_heads, head_size), input_dtype
        ),
        key: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, sequence_length, num_heads, head_size), input_dtype
        ),
        receptance: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, sequence_length, num_heads, head_size), input_dtype
        ),
        next_state: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, num_heads, head_size, head_size), "float32"
        ),
        mixed: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, sequence_length, num_heads, head_size), input_dtype
        ),
    ):
        with T.Kernel(batch_size, num_heads, threads=256) as (batch, head):
            current = T.alloc_shared((head_size, head_size), "float32")
            following = T.alloc_shared((head_size, head_size), "float32")
            for row, column in T.Parallel(head_size, head_size):
                current[row, column] = state[batch, head, row, column]
            T.sync_threads()

            for token in T.serial(sequence_length):
                for row, column in T.Parallel(head_size, head_size):
                    anti_projection = T.alloc_local((1,), "float32")
                    anti_projection[0] = 0.0
                    for inner in T.serial(head_size):
                        left = T.cast(
                            -normalized_key[batch, token, head, inner], input_dtype
                        )
                        right = T.cast(
                            normalized_key[batch, token, head, column]
                            * gate_a[batch, token, head, column],
                            input_dtype,
                        )
                        anti_element = T.cast(left * right, input_dtype)
                        anti_projection[0] += current[row, inner] * T.cast(
                            anti_element, "float32"
                        )
                    value_key = T.cast(
                        T.cast(
                            value[batch, token, head, row]
                            * key[batch, token, head, column],
                            input_dtype,
                        ),
                        "float32",
                    )
                    following[row, column] = (
                        current[row, column]
                        * T.cast(decay[batch, token, head, column], "float32")
                        + anti_projection[0]
                        + value_key
                    )
                T.sync_threads()
                for row, column in T.Parallel(head_size, head_size):
                    current[row, column] = following[row, column]
                T.sync_threads()

                for row in T.Parallel(head_size):
                    projection = T.alloc_local((1,), "float32")
                    projection[0] = 0.0
                    for column in T.serial(head_size):
                        projection[0] += (
                            T.cast(current[row, column], input_dtype)
                            * receptance[batch, token, head, column]
                        )
                    mixed[batch, token, head, row] = T.cast(projection[0], input_dtype)
                T.sync_threads()

            for row, column in T.Parallel(head_size, head_size):
                next_state[batch, head, row, column] = current[row, column]

    return kernel


@lru_cache(maxsize=32)
def _compiled_fast_state_scan(
    batch_size: int,
    sequence_length: int,
    num_heads: int,
    head_size: int,
    input_dtype: str,
    device_arch: str,
):
    del device_arch
    import tilelang  # type: ignore[import-not-found]

    return tilelang.compile(
        build_fast_state_scan_program(
            batch_size,
            sequence_length,
            num_heads,
            head_size,
            input_dtype,
        ),
        out_idx=[-2, -1],
        execution_backend="auto",
    )


def tilelang_fast_state_scan(
    state: torch.Tensor,
    decay: torch.Tensor,
    normalized_key: torch.Tensor,
    gate_a: torch.Tensor,
    value: torch.Tensor,
    key: torch.Tensor,
    receptance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run experimental approximate persistent sequence scan."""
    if state.device.type != "cuda":
        raise RuntimeError("TileLang fast state scan requires CUDA tensors")
    batch_size, sequence_length, num_heads, head_size = decay.shape
    kernel: Any = _compiled_fast_state_scan(
        batch_size,
        sequence_length,
        num_heads,
        head_size,
        _dtype_name(receptance.dtype),
        cuda_arch_key(state.device),
    )
    return kernel(
        state.contiguous(),
        decay.contiguous(),
        normalized_key.contiguous(),
        gate_a.contiguous(),
        value.contiguous(),
        key.contiguous(),
        receptance.contiguous(),
    )


def clear_tilelang_state_kernel_caches() -> None:
    """Drop bounded Python references to compiled recurrent-state kernels."""
    for compiler in (
        _compiled_kernel,
        _compiled_backward_kernel,
        _compiled_low_precision_state_kernel,
        _compiled_x_mix,
        _compiled_post_state,
        _compiled_fused_state_update,
        _compiled_state_projection,
        _compiled_fast_state_scan,
    ):
        compiler.cache_clear()
