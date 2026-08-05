from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch

from .kernel_tilelang_state import cuda_arch_key

_HEAD_SIZE = 64
_TWO_NEG_41 = 4.547473508864641e-13
_NEXP_HALF_LOG2_E = -0.8750387749145276
_NLOG2_E = -1.4426950408889634
_ROTATOR1_SIGNED = -1640531527


def _build_wkv_program(
    batch_size: int,
    sequence_length: int,
    num_heads: int,
    precise: bool = False,
):
    """Build a fixed-shape FP16 WKV scan with selectable recurrence precision."""
    import tilelang.language as T  # type: ignore[import-not-found]

    head_size = _HEAD_SIZE
    current_dtype = "float32" if precise else "float16"

    @T.prim_func
    def kernel(
        state: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, num_heads, head_size, head_size), "float16"
        ),
        receptance: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, sequence_length, num_heads, head_size), "float16"
        ),
        decay_raw: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, sequence_length, num_heads, head_size), "float16"
        ),
        key: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, sequence_length, num_heads, head_size), "float16"
        ),
        value: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, sequence_length, num_heads, head_size), "float16"
        ),
        gate_a: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, sequence_length, num_heads, head_size), "float16"
        ),
        gate_b: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, sequence_length, num_heads, head_size), "float16"
        ),
        elapsed: T.Tensor((batch_size,), "int32"),  # type: ignore[reportInvalidTypeForm]
        output: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (batch_size, sequence_length, num_heads, head_size), "float16"
        ),
    ):
        with T.Kernel(batch_size, num_heads, threads=head_size) as (batch, head):
            current = T.alloc_local((head_size,), current_dtype)
            r_shared = T.alloc_shared((2, head_size), "float16")
            w_shared = T.alloc_shared((2, head_size), "float16")
            k_shared = T.alloc_shared((2, head_size), "float16")
            v_shared = T.alloc_shared((2, head_size), "float16")
            a_shared = T.alloc_shared((2, head_size), "float16")
            b_shared = T.alloc_shared((2, head_size), "float16")

            row = T.get_thread_binding()
            for column in T.serial(head_size):
                current[column] = T.cast(
                    state[batch, head, row, column], current_dtype
                )

            T.async_copy(
                receptance[batch, 0, head, 0:head_size],
                r_shared[0, 0:head_size],
                coalesced_width=8,
            )
            T.async_copy(
                decay_raw[batch, 0, head, 0:head_size],
                w_shared[0, 0:head_size],
                coalesced_width=8,
            )
            T.async_copy(
                key[batch, 0, head, 0:head_size],
                k_shared[0, 0:head_size],
                coalesced_width=8,
            )
            T.async_copy(
                value[batch, 0, head, 0:head_size],
                v_shared[0, 0:head_size],
                coalesced_width=8,
            )
            T.async_copy(
                gate_a[batch, 0, head, 0:head_size],
                a_shared[0, 0:head_size],
                coalesced_width=8,
            )
            T.async_copy(
                gate_b[batch, 0, head, 0:head_size],
                b_shared[0, 0:head_size],
                coalesced_width=8,
            )

            for token in T.serial(sequence_length):
                current_buffer = token % 2
                T.ptx_wait_group(0)
                T.sync_threads()
                if not precise:
                    for column in T.Parallel(head_size):
                        raw = T.cast(
                            w_shared[current_buffer, column], "float32"
                        )
                        phase = T.cast(
                            elapsed[batch] + head * head_size + column + token,
                            "int32",
                        )
                        rotation = T.cast(
                            phase * T.cast(_ROTATOR1_SIGNED, "int32"), "float32"
                        ) * _TWO_NEG_41
                        transformed = T.exp2(
                            _NEXP_HALF_LOG2_E
                            / (1.0 + T.exp2(_NLOG2_E * raw))
                        ) - 1.0 + rotation
                        w_shared[current_buffer, column] = T.cast(
                            transformed, "float16"
                        )
                    T.sync_threads()

                if token + 1 < sequence_length:
                    next_buffer = (token + 1) % 2
                    T.async_copy(
                        receptance[batch, token + 1, head, 0:head_size],
                        r_shared[next_buffer, 0:head_size],
                        coalesced_width=8,
                    )
                    T.async_copy(
                        decay_raw[batch, token + 1, head, 0:head_size],
                        w_shared[next_buffer, 0:head_size],
                        coalesced_width=8,
                    )
                    T.async_copy(
                        key[batch, token + 1, head, 0:head_size],
                        k_shared[next_buffer, 0:head_size],
                        coalesced_width=8,
                    )
                    T.async_copy(
                        value[batch, token + 1, head, 0:head_size],
                        v_shared[next_buffer, 0:head_size],
                        coalesced_width=8,
                    )
                    T.async_copy(
                        gate_a[batch, token + 1, head, 0:head_size],
                        a_shared[next_buffer, 0:head_size],
                        coalesced_width=8,
                    )
                    T.async_copy(
                        gate_b[batch, token + 1, head, 0:head_size],
                        b_shared[next_buffer, 0:head_size],
                        coalesced_width=8,
                    )

                if precise:
                    # Match the PyTorch recurrent store boundary: accumulate the
                    # factored rank-one update in FP32, then round state once.
                    state_projection_f32 = T.alloc_local((1,), "float32")
                    state_projection_f32[0] = T.cast(0.0, "float32")
                    for column in T.serial(head_size):
                        state_projection_f32[0] += current[column] * T.cast(
                            a_shared[current_buffer, column], "float32"
                        )
                    rounded_state = T.alloc_local((head_size,), "float16")
                    for column in T.serial(head_size):
                        updated_f32 = T.alloc_local((1,), "float32")
                        updated_f32[0] = current[column] * T.cast(
                            w_shared[current_buffer, column], "float32"
                        )
                        updated_f32[0] = (
                            updated_f32[0]
                            + state_projection_f32[0]
                            * T.cast(b_shared[current_buffer, column], "float32")
                        )
                        value_key = T.cast(
                            k_shared[current_buffer, column]
                            * v_shared[current_buffer, row],
                            "float16",
                        )
                        updated_f32[0] = updated_f32[0] + T.cast(
                            value_key, "float32"
                        )
                        rounded = T.cast(updated_f32[0], "float16")
                        current[column] = T.cast(rounded, "float32")
                        rounded_state[column] = rounded
                    # Fixed two-lane FP16 projection preserves recurrent output
                    # order while avoiding a second long FP32 dependency chain.
                    output_pair = T.alloc_local((2,), "float16")
                    for lane in T.vectorized(2):
                        output_pair[lane] = T.cast(0.0, "float16")
                    for pair in T.serial(head_size // 2):
                        for lane in T.vectorized(2):
                            column = pair * 2 + lane
                            output_pair[lane] = T.cast(
                                rounded_state[column]
                                * r_shared[current_buffer, column]
                                + output_pair[lane],
                                "float16",
                            )
                    output[batch, token, head, row] = T.cast(
                        output_pair[0] + output_pair[1], "float16"
                    )
                else:
                    projection_pair = T.alloc_local((2,), "float16")
                    for lane in T.vectorized(2):
                        projection_pair[lane] = T.cast(0.0, "float16")
                    for pair in T.serial(head_size // 2):
                        for lane in T.vectorized(2):
                            column = pair * 2 + lane
                            projection_pair[lane] = T.cast(
                                a_shared[current_buffer, column] * current[column]
                                + projection_pair[lane],
                                "float16",
                            )
                    state_projection = T.cast(
                        projection_pair[0] + projection_pair[1], "float16"
                    )
                    output_pair = T.alloc_local((2,), "float16")
                    for lane in T.vectorized(2):
                        output_pair[lane] = T.cast(0.0, "float16")
                    for pair in T.serial(head_size // 2):
                        for lane in T.vectorized(2):
                            column = pair * 2 + lane
                            updated = T.cast(
                                current[column]
                                * w_shared[current_buffer, column]
                                + T.cast(
                                    k_shared[current_buffer, column]
                                    * v_shared[current_buffer, row]
                                    + T.cast(
                                        state_projection
                                        * b_shared[current_buffer, column]
                                        + current[column],
                                        "float16",
                                    ),
                                    "float16",
                                ),
                                "float16",
                            )
                            current[column] = updated
                            output_pair[lane] = T.cast(
                                updated * r_shared[current_buffer, column]
                                + output_pair[lane],
                                "float16",
                            )
                    output[batch, token, head, row] = T.cast(
                        output_pair[0] + output_pair[1], "float16"
                    )
                # Precise mode synchronizes at the next wait/barrier before a
                # double buffer is reused; the legacy transform needs this barrier.
                if not precise:
                    T.sync_threads()

            for column in T.serial(head_size):
                state[batch, head, row, column] = current[column]

    return kernel




@lru_cache(maxsize=16)
def _compiled_wkv(
    batch_size: int,
    sequence_length: int,
    num_heads: int,
    device_arch: str,
    precise: bool = False,
):
    del device_arch
    import tilelang  # type: ignore[import-not-found]

    return tilelang.compile(
        _build_wkv_program(
            batch_size,
            sequence_length,
            num_heads,
            precise,
        ),
        out_idx=[],
        execution_backend="auto",
    )


def _build_wkv_w0_t1_program(num_heads: int):
    """Build specialized B1T1 FP16 WKV with fused decay bias."""
    import tilelang.language as T  # type: ignore[import-not-found]

    head_size = _HEAD_SIZE

    @T.prim_func
    def kernel(
        state: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (1, num_heads, head_size, head_size), "float16"
        ),
        receptance: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (1, 1, num_heads, head_size), "float16"
        ),
        decay: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (1, 1, num_heads, head_size), "float16"
        ),
        decay_bias: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (num_heads, head_size), "float16"
        ),
        key: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (1, 1, num_heads, head_size), "float16"
        ),
        value: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (1, 1, num_heads, head_size), "float16"
        ),
        gate_a: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (1, 1, num_heads, head_size), "float16"
        ),
        gate_b: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (1, 1, num_heads, head_size), "float16"
        ),
        elapsed: T.Tensor((1,), "int32"),  # type: ignore[reportInvalidTypeForm]
        output: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (1, 1, num_heads, head_size), "float16"
        ),
    ):
        with T.Kernel(num_heads, threads=head_size) as head:
            row = T.get_thread_binding(0)
            current = T.alloc_local((head_size,), "float16")
            state_vector = T.alloc_local((8,), "float16")
            state_shared = T.alloc_shared(
                (head_size, head_size), "float16"
            )
            r_shared = T.alloc_shared((head_size,), "float16")
            w_shared = T.alloc_shared((head_size,), "float16")
            k_shared = T.alloc_shared((head_size,), "float16")
            v_shared = T.alloc_shared((head_size,), "float16")
            a_shared = T.alloc_shared((head_size,), "float16")
            b_shared = T.alloc_shared((head_size,), "float16")
            for chunk in T.serial(head_size // 8):
                linear_start = (chunk * head_size + row) * 8
                for vector_lane in T.vectorized(8):
                    linear = linear_start + vector_lane
                    source_row = linear // head_size
                    source_column = linear % head_size
                    state_vector[vector_lane] = state[
                        0, head, source_row, source_column
                    ]
                for vector_lane in T.serial(8):
                    linear = linear_start + vector_lane
                    source_row = linear // head_size
                    source_column = linear % head_size
                    swizzled_column = (
                        ((source_row % 32) ^ (source_column // 2)) * 2
                        + source_column % 2
                    )
                    state_shared[source_row, swizzled_column] = (
                        state_vector[vector_lane]
                    )
            T.sync_threads()
            for column in T.serial(head_size):
                swizzled_column = (
                    ((row % 32) ^ (column // 2)) * 2 + column % 2
                )
                current[column] = state_shared[row, swizzled_column]
            r_shared[row] = receptance[0, 0, head, row]
            k_shared[row] = key[0, 0, head, row]
            v_shared[row] = value[0, 0, head, row]
            a_shared[row] = gate_a[0, 0, head, row]
            b_shared[row] = gate_b[0, 0, head, row]
            raw = T.cast(
                decay[0, 0, head, row] + decay_bias[head, row],
                "float32",
            )
            phase = T.cast(
                elapsed[0] + head * head_size + row, "int32"
            )
            rotation = T.cast(
                phase * T.cast(_ROTATOR1_SIGNED, "int32"), "float32"
            ) * _TWO_NEG_41
            transformed = T.exp2(
                _NEXP_HALF_LOG2_E
                / (1.0 + T.exp2(_NLOG2_E * raw))
            ) - 1.0 + rotation
            w_shared[row] = T.cast(transformed, "float16")
            T.sync_threads()
            projection_pair = T.alloc_local((2,), "float16")
            for pair_lane in T.vectorized(2):
                projection_pair[pair_lane] = T.cast(0.0, "float16")
            for pair in T.serial(head_size // 2):
                for pair_lane in T.vectorized(2):
                    column = pair * 2 + pair_lane
                    projection_pair[pair_lane] = T.cast(
                        a_shared[column] * current[column]
                        + projection_pair[pair_lane],
                        "float16",
                    )
            state_projection = T.cast(
                projection_pair[0] + projection_pair[1], "float16"
            )
            output_pair = T.alloc_local((2,), "float16")
            for pair_lane in T.vectorized(2):
                output_pair[pair_lane] = T.cast(0.0, "float16")
            for pair in T.serial(head_size // 2):
                for pair_lane in T.vectorized(2):
                    column = pair * 2 + pair_lane
                    updated = T.cast(
                        current[column] * w_shared[column]
                        + T.cast(
                            k_shared[column] * v_shared[row]
                            + T.cast(
                                state_projection * b_shared[column]
                                + current[column],
                                "float16",
                            ),
                            "float16",
                        ),
                        "float16",
                    )
                    current[column] = updated
                    output_pair[pair_lane] = T.cast(
                        updated * r_shared[column]
                        + output_pair[pair_lane],
                        "float16",
                    )
            output[0, 0, head, row] = T.cast(
                output_pair[0] + output_pair[1], "float16"
            )
            for column in T.serial(head_size):
                swizzled_column = (
                    ((row % 32) ^ (column // 2)) * 2 + column % 2
                )
                state_shared[row, swizzled_column] = current[column]
            T.sync_threads()
            for chunk in T.serial(head_size // 8):
                linear_start = (chunk * head_size + row) * 8
                for vector_lane in T.serial(8):
                    linear = linear_start + vector_lane
                    target_row = linear // head_size
                    target_column = linear % head_size
                    swizzled_column = (
                        ((target_row % 32) ^ (target_column // 2)) * 2
                        + target_column % 2
                    )
                    state_vector[vector_lane] = state_shared[
                        target_row, swizzled_column
                    ]
                for vector_lane in T.vectorized(8):
                    linear = linear_start + vector_lane
                    target_row = linear // head_size
                    target_column = linear % head_size
                    state[0, head, target_row, target_column] = (
                        state_vector[vector_lane]
                    )

    return kernel


@lru_cache(maxsize=8)
def _compiled_wkv_w0_t1(num_heads: int, device_arch: str):
    del device_arch
    import tilelang  # type: ignore[import-not-found]

    return tilelang.compile(
        _build_wkv_w0_t1_program(num_heads),
        out_idx=[],
        execution_backend="auto",
    )


def _wkv_w0_t1_out(
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
    """Run specialized B1T1 WKV with fused decay bias."""
    batch, sequence, heads, head_size = receptance.shape
    if (batch, sequence, head_size) != (1, 1, _HEAD_SIZE):
        raise ValueError("fused-bias WKV requires B1T1 with head size 64")
    tensors = (
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
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise RuntimeError("fused-bias WKV requires CUDA")
    if any(tensor.dtype != torch.float16 for tensor in tensors):
        raise TypeError("fused-bias WKV requires FP16")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("fused-bias WKV requires contiguous tensors")
    if tuple(state.shape) != (1, heads, head_size, head_size):
        raise ValueError("fused-bias WKV state shape mismatch")
    if tuple(decay_bias.shape) != (heads, head_size):
        raise ValueError("fused-bias WKV decay-bias shape mismatch")
    if any(
        tuple(tensor.shape) != tuple(receptance.shape)
        for tensor in (decay, key, value, gate_a, gate_b, output)
    ):
        raise ValueError("fused-bias WKV vector shape mismatch")
    if elapsed.dtype != torch.int32 or tuple(elapsed.shape) != (1,):
        raise TypeError("elapsed must be int32 [1]")
    kernel: Any = _compiled_wkv_w0_t1(
        heads, cuda_arch_key(state.device)
    )
    kernel(
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


def _wkv_kernel_out(
    state: torch.Tensor,
    receptance: torch.Tensor,
    decay: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate_a: torch.Tensor,
    gate_b: torch.Tensor,
    elapsed: torch.Tensor,
    output: torch.Tensor,
    *,
    precise: bool,
) -> None:
    if state.device.type != "cuda" or output.device.type != "cuda":
        raise RuntimeError("TileLang FP16 WKV requires CUDA tensors")
    batch_size, sequence_length, num_heads, head_size = receptance.shape
    expected_state = (batch_size, num_heads, head_size, head_size)
    if head_size != _HEAD_SIZE or tuple(state.shape) != expected_state:
        raise ValueError("TileLang WKV requires [B,H,64,64] state")
    if not state.is_contiguous() or not output.is_contiguous():
        raise ValueError("TileLang WKV state/output must be contiguous")
    vectors = (receptance, decay, key, value, gate_a, gate_b)
    if any(vector.shape != receptance.shape for vector in vectors):
        raise ValueError("All WKV vectors must have the same [B,T,H,N] shape")
    if output.shape != receptance.shape:
        raise ValueError("WKV output must match receptance shape")
    if any(vector.dtype != torch.float16 for vector in (*vectors, state, output)):
        raise TypeError("TileLang WKV tensors must use float16")
    if (
        elapsed.dtype != torch.int32
        or tuple(elapsed.shape) != (batch_size,)
        or not elapsed.is_contiguous()
    ):
        raise TypeError("elapsed must be contiguous int32 [B]")
    kernel: Any = _compiled_wkv(
        batch_size,
        sequence_length,
        num_heads,
        cuda_arch_key(state.device),
        precise,
    )
    kernel(
        state,
        receptance.contiguous(),
        decay.contiguous(),
        key.contiguous(),
        value.contiguous(),
        gate_a.contiguous(),
        gate_b.contiguous(),
        elapsed,
        output,
    )


def _wkv_out(
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
    """Run Albatross-compatible WKV into caller-owned state/output."""
    _wkv_kernel_out(
        state,
        receptance,
        decay_raw,
        key,
        value,
        gate_a,
        gate_b,
        elapsed,
        output,
        precise=False,
    )


def _wkv_precise_out(
    state: torch.Tensor,
    receptance: torch.Tensor,
    decay: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate_a: torch.Tensor,
    gate_b: torch.Tensor,
    elapsed: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Run FP32-accumulating WKV with FP16 recurrent storage."""
    _wkv_kernel_out(
        state,
        receptance,
        decay,
        key,
        value,
        gate_a,
        gate_b,
        elapsed,
        output,
        precise=True,
    )





def _build_gemv_program(
    input_rows: int,
    output_rows: int,
    input_dtype: str = "float16",
    out_tile: int = 2,
    reduce_threads: int = 128,
    clear_input_sized_output: bool = False,
):
    """Build output-tiled FP32-accumulating inference GEMV."""
    if input_rows <= 0 or output_rows <= 0 or output_rows % out_tile:
        raise ValueError("GEMV dimensions must be positive and output tiled")
    if input_dtype not in {"float16", "bfloat16"}:
        raise ValueError("GEMV dtype must be float16 or bfloat16")
    vector_width = 8
    block_k = reduce_threads * vector_width
    if input_rows % block_k:
        raise ValueError("GEMV input must divide reduction tile")

    import tilelang.language as T  # type: ignore[import-not-found]

    @T.prim_func
    def kernel(
        value: T.Tensor((input_rows,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        weight: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (output_rows, input_rows), input_dtype
        ),
        output: T.Tensor((output_rows,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        clear_output: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (input_rows,), input_dtype
        ),
    ):
        with T.Kernel(
            output_rows // out_tile, threads=reduce_threads
        ) as block:
            thread = T.get_thread_binding(0)
            row = block * out_tile
            value_local = T.alloc_local((vector_width,), input_dtype)
            weight_local = T.alloc_local(
                (out_tile, vector_width), input_dtype
            )
            accumulator0 = T.alloc_local((1,), "float32")
            accumulator1 = T.alloc_local((1,), "float32")
            reduced0 = T.alloc_local((1,), "float32")
            reduced1 = T.alloc_local((1,), "float32")
            T.clear(accumulator0)
            T.clear(accumulator1)
            for chunk in T.serial(input_rows // block_k):
                for lane in T.vectorized(vector_width):
                    column = (
                        chunk * block_k + thread * vector_width + lane
                    )
                    value_local[lane] = value[column]
                    weight_local[0, lane] = weight[row, column]
                    weight_local[1, lane] = weight[row + 1, column]
                for lane in T.serial(vector_width):
                    current = T.cast(value_local[lane], "float32")
                    accumulator0[0] += current * T.cast(
                        weight_local[0, lane], "float32"
                    )
                    accumulator1[0] += current * T.cast(
                        weight_local[1, lane], "float32"
                    )
            with T.attr(
                T.comm_reducer(
                    lambda left, right: left + right,
                    [T.cast(0, "float32")],
                ),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        accumulator0[0],
                        True,
                        reduced0[0],
                        thread,
                        dtype="handle",
                    )
                )
            with T.attr(
                T.comm_reducer(
                    lambda left, right: left + right,
                    [T.cast(0, "float32")],
                ),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        accumulator1[0],
                        True,
                        reduced1[0],
                        thread,
                        dtype="handle",
                    )
                )
            if thread == 0:
                output[row] = T.cast(reduced0[0], input_dtype)
                output[row + 1] = T.cast(reduced1[0], input_dtype)
                if clear_input_sized_output and row < input_rows:
                    clear_output[row] = T.cast(0.0, input_dtype)
                    clear_output[row + 1] = T.cast(0.0, input_dtype)

    return kernel


@lru_cache(maxsize=32)
def _compiled_gemv(
    input_rows: int,
    output_rows: int,
    input_dtype: str,
    out_tile: int,
    reduce_threads: int,
    clear_input_sized_output: bool,
    device_arch: str,
):
    del device_arch
    import tilelang  # type: ignore[import-not-found]

    return tilelang.compile(
        _build_gemv_program(
            input_rows,
            output_rows,
            input_dtype,
            out_tile,
            reduce_threads,
            clear_input_sized_output,
        ),
        out_idx=[],
        execution_backend="auto",
    )

def _build_ffn_program(
    channels: int,
    ffn_rows: int,
    block_rows: int = 2,
    reduce_threads: int = 64,
    input_dtype: str = "float16",
):
    """Build B1T1 FFN key GEMV fused with ReLU-square."""
    if channels <= 0 or ffn_rows <= 0 or ffn_rows % block_rows:
        raise ValueError("dimensions must be positive and rows divisible by block_rows")
    if input_dtype not in {"float16", "bfloat16"}:
        raise ValueError("FFN dtype must be float16 or bfloat16")
    vector_width = 8
    block_k = reduce_threads * vector_width
    if channels % block_k:
        raise ValueError("channels must be divisible by reduce_threads * 8")

    import tilelang.language as T  # type: ignore[import-not-found]

    @T.prim_func
    def kernel(
        mixed: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        weight: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (ffn_rows, channels), input_dtype
        ),
        output: T.Tensor((ffn_rows,), input_dtype),  # type: ignore[reportInvalidTypeForm]
    ):
        with T.Kernel(
            ffn_rows // block_rows,
            threads=(block_rows, reduce_threads),
        ) as block:
            row_lane = T.get_thread_binding(0)
            reduce_lane = T.get_thread_binding(1)
            row = block * block_rows + row_lane
            mixed_local = T.alloc_local((vector_width,), input_dtype)
            weight_local = T.alloc_local((vector_width,), input_dtype)
            accumulator = T.alloc_local((1,), "float32")
            reduced = T.alloc_local((1,), "float32")
            T.clear(accumulator)
            for chunk in T.serial(channels // block_k):
                for lane in T.vectorized(vector_width):
                    column = (
                        chunk * block_k
                        + reduce_lane * vector_width
                        + lane
                    )
                    mixed_local[lane] = mixed[column]
                    weight_local[lane] = weight[row, column]
                for lane in T.serial(vector_width):
                    accumulator[0] += T.cast(
                        mixed_local[lane], "float32"
                    ) * T.cast(weight_local[lane], "float32")
            with T.attr(
                T.comm_reducer(
                    lambda left, right: left + right,
                    [T.cast(0, "float32")],
                ),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        accumulator[0],
                        True,
                        reduced[0],
                        reduce_lane,
                        dtype="handle",
                    )
                )
            if reduce_lane == 0:
                projected = T.cast(reduced[0], input_dtype)
                activated = T.max(projected, T.cast(0, input_dtype))
                output[row] = T.cast(activated * activated, input_dtype)

    return kernel


@lru_cache(maxsize=32)
def _compiled_ffn(
    channels: int,
    ffn_rows: int,
    block_rows: int,
    reduce_threads: int,
    input_dtype: str,
    device_arch: str,
):
    del device_arch
    import tilelang  # type: ignore[import-not-found]

    return tilelang.compile(
        _build_ffn_program(
            channels, ffn_rows, block_rows, reduce_threads, input_dtype
        ),
        out_idx=[],
        execution_backend="auto",
    )


def _build_tmix_layernorm_mix6_program(
    channels: int,
    input_dtype: str = "float16",
    epsilon: float = 1e-5,
    threads: int = 256,
):
    """Build fused LayerNorm and six shifted time-mix vectors."""
    if channels <= 0 or channels % threads:
        raise ValueError("channels must be divisible by LayerNorm threads")
    if input_dtype not in {"float16", "bfloat16"}:
        raise ValueError("time-mix dtype must be float16 or bfloat16")

    import tilelang.language as T  # type: ignore[import-not-found]

    @T.prim_func
    def kernel(
        residual: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        previous: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        norm_weight: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        norm_bias: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        mix_weights: T.Tensor((6, channels), input_dtype),  # type: ignore[reportInvalidTypeForm]
        normalized: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        mixed: T.Tensor((6, channels), input_dtype),  # type: ignore[reportInvalidTypeForm]
    ):
        with T.Kernel(1, threads=threads):
            thread = T.get_thread_binding(0)
            local_sum = T.alloc_local((1,), "float32")
            local_square = T.alloc_local((1,), "float32")
            reduced_sum = T.alloc_local((1,), "float32")
            reduced_square = T.alloc_local((1,), "float32")
            T.clear(local_sum)
            T.clear(local_square)
            for chunk in T.serial(channels // threads):
                channel = chunk * threads + thread
                value = T.cast(residual[channel], "float32")
                local_sum[0] += value
                local_square[0] += value * value
            with T.attr(
                T.comm_reducer(
                    lambda left, right: left + right,
                    [T.cast(0, "float32")],
                ),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        local_sum[0],
                        True,
                        reduced_sum[0],
                        thread,
                        dtype="handle",
                    )
                )
            with T.attr(
                T.comm_reducer(
                    lambda left, right: left + right,
                    [T.cast(0, "float32")],
                ),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        local_square[0],
                        True,
                        reduced_square[0],
                        thread,
                        dtype="handle",
                    )
                )
            mean = reduced_sum[0] / channels
            variance = T.max(
                reduced_square[0] / channels - mean * mean, 0.0
            )
            inverse_std = 1.0 / T.sqrt(variance + epsilon)
            for chunk in T.serial(channels // threads):
                channel = chunk * threads + thread
                current = T.cast(
                    (
                        (T.cast(residual[channel], "float32") - mean)
                        * inverse_std
                        * T.cast(norm_weight[channel], "float32")
                        + T.cast(norm_bias[channel], "float32")
                    ),
                    input_dtype,
                )
                previous_value = previous[channel]
                normalized[channel] = current
                delta = T.cast(previous_value - current, input_dtype)
                for stream in T.serial(6):
                    mixed[stream, channel] = T.cast(
                        current + delta * mix_weights[stream, channel],
                        input_dtype,
                    )

    return kernel


@lru_cache(maxsize=32)
def _compiled_tmix_layernorm_mix6(
    channels: int,
    input_dtype: str,
    epsilon: float,
    threads: int,
    device_arch: str,
):
    del device_arch
    import tilelang  # type: ignore[import-not-found]

    return tilelang.compile(
        _build_tmix_layernorm_mix6_program(
            channels, input_dtype, epsilon, threads
        ),
        out_idx=[],
        execution_backend="auto",
    )

def _build_cmix_layernorm_mix_program(
    channels: int,
    input_dtype: str = "float16",
    epsilon: float = 1e-5,
    threads: int = 256,
):
    """Build fused final residual LayerNorm and channel-mix input."""
    if channels <= 0 or channels % threads:
        raise ValueError("channels must be divisible by LayerNorm threads")
    if input_dtype not in {"float16", "bfloat16"}:
        raise ValueError("channel-mix dtype must be float16 or bfloat16")

    import tilelang.language as T  # type: ignore[import-not-found]

    @T.prim_func
    def kernel(
        residual: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        previous: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        norm_weight: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        norm_bias: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        mix_weight: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        normalized: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        mixed: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
    ):
        with T.Kernel(1, threads=threads):
            thread = T.get_thread_binding(0)
            local_sum = T.alloc_local((1,), "float32")
            local_square = T.alloc_local((1,), "float32")
            reduced_sum = T.alloc_local((1,), "float32")
            reduced_square = T.alloc_local((1,), "float32")
            T.clear(local_sum)
            T.clear(local_square)
            for chunk in T.serial(channels // threads):
                channel = chunk * threads + thread
                value = T.cast(residual[channel], "float32")
                local_sum[0] += value
                local_square[0] += value * value
            with T.attr(
                T.comm_reducer(
                    lambda left, right: left + right,
                    [T.cast(0, "float32")],
                ),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        local_sum[0],
                        True,
                        reduced_sum[0],
                        thread,
                        dtype="handle",
                    )
                )
            with T.attr(
                T.comm_reducer(
                    lambda left, right: left + right,
                    [T.cast(0, "float32")],
                ),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        local_square[0],
                        True,
                        reduced_square[0],
                        thread,
                        dtype="handle",
                    )
                )
            mean = reduced_sum[0] / channels
            variance = T.max(
                reduced_square[0] / channels - mean * mean, 0.0
            )
            inverse_std = 1.0 / T.sqrt(variance + epsilon)
            for chunk in T.serial(channels // threads):
                channel = chunk * threads + thread
                current = T.cast(
                    (
                        (T.cast(residual[channel], "float32") - mean)
                        * inverse_std
                        * T.cast(norm_weight[channel], "float32")
                        + T.cast(norm_bias[channel], "float32")
                    ),
                    input_dtype,
                )
                normalized[channel] = current
                delta = T.cast(previous[channel] - current, input_dtype)
                mixed[channel] = T.cast(
                    current + delta * mix_weight[channel], input_dtype
                )

    return kernel


def _build_cmix_add_layernorm_mix_program(
    channels: int,
    input_dtype: str = "float16",
    epsilon: float = 1e-5,
    threads: int = 256,
):
    """Build fused residual add, LayerNorm, and channel-mix input."""
    if channels <= 0 or channels % threads:
        raise ValueError("channels must be divisible by LayerNorm threads")
    if input_dtype not in {"float16", "bfloat16"}:
        raise ValueError("channel-mix dtype must be float16 or bfloat16")

    import tilelang.language as T  # type: ignore[import-not-found]

    @T.prim_func
    def kernel(
        residual: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        update: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        previous: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        norm_weight: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        norm_bias: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        mix_weight: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        combined: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        normalized: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        mixed: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
    ):
        with T.Kernel(1, threads=threads):
            thread = T.get_thread_binding(0)
            local_sum = T.alloc_local((1,), "float32")
            local_square = T.alloc_local((1,), "float32")
            reduced_sum = T.alloc_local((1,), "float32")
            reduced_square = T.alloc_local((1,), "float32")
            T.clear(local_sum)
            T.clear(local_square)
            for chunk in T.serial(channels // threads):
                channel = chunk * threads + thread
                value_f16 = T.cast(
                    residual[channel] + update[channel], input_dtype
                )
                combined[channel] = value_f16
                value = T.cast(value_f16, "float32")
                local_sum[0] += value
                local_square[0] += value * value
            with T.attr(
                T.comm_reducer(
                    lambda left, right: left + right,
                    [T.cast(0, "float32")],
                ),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        local_sum[0],
                        True,
                        reduced_sum[0],
                        thread,
                        dtype="handle",
                    )
                )
            with T.attr(
                T.comm_reducer(
                    lambda left, right: left + right,
                    [T.cast(0, "float32")],
                ),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        local_square[0],
                        True,
                        reduced_square[0],
                        thread,
                        dtype="handle",
                    )
                )
            mean = reduced_sum[0] / channels
            variance = T.max(
                reduced_square[0] / channels - mean * mean, 0.0
            )
            inverse_std = 1.0 / T.sqrt(variance + epsilon)
            for chunk in T.serial(channels // threads):
                channel = chunk * threads + thread
                current = T.cast(
                    (
                        (T.cast(combined[channel], "float32") - mean)
                        * inverse_std
                        * T.cast(norm_weight[channel], "float32")
                        + T.cast(norm_bias[channel], "float32")
                    ),
                    input_dtype,
                )
                previous_value = previous[channel]
                normalized[channel] = current
                delta = T.cast(previous_value - current, input_dtype)
                mixed[channel] = T.cast(
                    current + delta * mix_weight[channel], input_dtype
                )

    return kernel


@lru_cache(maxsize=32)
def _compiled_cmix_add_layernorm_mix(
    channels: int,
    input_dtype: str,
    epsilon: float,
    threads: int,
    device_arch: str,
):
    del device_arch
    import tilelang  # type: ignore[import-not-found]

    return tilelang.compile(
        _build_cmix_add_layernorm_mix_program(
            channels, input_dtype, epsilon, threads
        ),
        out_idx=[],
        execution_backend="auto",
    )


@lru_cache(maxsize=32)
def _compiled_cmix_layernorm_mix(
    channels: int,
    input_dtype: str,
    epsilon: float,
    threads: int,
    device_arch: str,
):
    del device_arch
    import tilelang  # type: ignore[import-not-found]

    return tilelang.compile(
        _build_cmix_layernorm_mix_program(
            channels, input_dtype, epsilon, threads
        ),
        out_idx=[],
        execution_backend="auto",
    )


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    raise TypeError("TileLang decode tensors must use float16 or bfloat16")


def _require_contiguous_cuda(
    tensors: tuple[torch.Tensor, ...], name: str
) -> str:
    if torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in tensors
    ):
        raise RuntimeError(f"TileLang {name} is inference-only")
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise RuntimeError(f"TileLang {name} requires CUDA tensors")
    dtype = tensors[0].dtype
    input_dtype = _dtype_name(dtype)
    if any(tensor.dtype != dtype for tensor in tensors):
        raise TypeError(f"TileLang {name} tensors must share one dtype")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError(f"TileLang {name} requires contiguous tensors")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError(f"TileLang {name} tensors must share a device")
    return input_dtype


def _ffn_out(
    mixed: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    *,
    block_rows: int = 1,
    reduce_threads: int = 128,
) -> None:
    """Run FFN key projection and ReLU-square into caller-owned output."""
    input_dtype = _require_contiguous_cuda(
        (mixed, weight, output), "FFN key"
    )
    if mixed.dim() != 1 or weight.dim() != 2:
        raise ValueError("FFN key expects mixed [C] and weight [F,C]")
    channels = mixed.numel()
    ffn_rows = weight.size(0)
    if weight.size(1) != channels or tuple(output.shape) != (ffn_rows,):
        raise ValueError("FFN key weight/output shape mismatch")
    kernel: Any = _compiled_ffn(
        channels,
        ffn_rows,
        block_rows,
        reduce_threads,
        input_dtype,
        cuda_arch_key(mixed.device),
    )
    kernel(mixed, weight, output)


def _build_cmix_value_program(
    channels: int,
    ffn_rows: int,
    input_dtype: str = "float16",
    block_rows: int = 64,
    reduce_threads: int = 8,
):
    """Build sparse FFN value projection over transposed packed weights."""
    if channels <= 0 or ffn_rows <= 0:
        raise ValueError("dimensions must be positive")
    if channels % block_rows or ffn_rows % reduce_threads:
        raise ValueError("dimensions must divide block and reduction sizes")
    if input_dtype not in {"float16", "bfloat16"}:
        raise ValueError("FFN dtype must be float16 or bfloat16")

    import tilelang.language as T  # type: ignore[import-not-found]

    @T.prim_func
    def kernel(
        hidden: T.Tensor((ffn_rows,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        weight: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (ffn_rows, channels), input_dtype
        ),
        output: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
    ):
        with T.Kernel(
            channels // block_rows,
            threads=(reduce_threads, block_rows),
        ) as block:
            reduce_lane = T.get_thread_binding(0)
            output_lane = T.get_thread_binding(1)
            row = block * block_rows + output_lane
            accumulator = T.alloc_local((1,), "float32")
            reduced = T.alloc_local((1,), "float32")
            T.clear(accumulator)
            for chunk in T.serial(ffn_rows // reduce_threads):
                hidden_row = chunk * reduce_threads + reduce_lane
                activation = hidden[hidden_row]
                if activation != T.cast(0, input_dtype):
                    accumulator[0] += T.cast(
                        activation, "float32"
                    ) * T.cast(weight[hidden_row, row], "float32")
            with T.attr(
                T.comm_reducer(
                    lambda left, right: left + right,
                    [T.cast(0, "float32")],
                ),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        accumulator[0],
                        True,
                        reduced[0],
                        reduce_lane,
                        dtype="handle",
                    )
                )
            if reduce_lane == 0:
                output[row] = T.cast(reduced[0], input_dtype)

    return kernel


@lru_cache(maxsize=32)
def _compiled_cmix_value(
    channels: int,
    ffn_rows: int,
    input_dtype: str,
    block_rows: int,
    reduce_threads: int,
    device_arch: str,
):
    del device_arch
    import tilelang  # type: ignore[import-not-found]

    return tilelang.compile(
        _build_cmix_value_program(
            channels, ffn_rows, input_dtype, block_rows, reduce_threads
        ),
        out_idx=[],
        execution_backend="auto",
    )


def _build_cmix_sparse_atomic_program(
    channels: int,
    ffn_rows: int,
    input_dtype: str = "float16",
    ffn_tile: int = 128,
    threads: int = 128,
):
    """Build tiled exact-zero FFN down projection with FP16 atomics."""
    output_tile = threads * 2
    if (
        channels <= 0
        or ffn_rows <= 0
        or ffn_rows % ffn_tile
        or channels % output_tile
    ):
        raise ValueError("FFN dimensions must divide sparse tiles")
    if input_dtype != "float16":
        raise ValueError("atomic sparse FFN currently requires float16")

    import tilelang.language as T  # type: ignore[import-not-found]

    @T.prim_func
    def kernel(
        hidden: T.Tensor((ffn_rows,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        weight: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (ffn_rows, channels), input_dtype
        ),
        output: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
    ):
        with T.Kernel(
            ffn_rows // ffn_tile,
            channels // output_tile,
            threads=threads,
        ) as (ffn_block, channel_block):
            thread = T.get_thread_binding(0)
            activation_shared = T.alloc_shared((ffn_tile,), input_dtype)
            nonzero_ids = T.alloc_shared((ffn_tile,), "int32")
            nonzero_count = T.alloc_shared((1,), "int32")
            warp_counts = T.alloc_shared((ffn_tile // 32,), "int32")
            warp_prefix = T.alloc_shared((ffn_tile // 32,), "int32")
            accumulators = T.alloc_local((2,), input_dtype)
            ffn_row = ffn_block * ffn_tile + thread
            preactivation = T.cast(hidden[ffn_row], "float32")
            activated = T.max(preactivation, 0.0)
            activation_shared[thread] = T.cast(
                activated * activated, input_dtype
            )
            if thread == 0:
                nonzero_count[0] = 0
            T.sync_threads()
            lane = thread % 32
            warp = thread // 32
            nonzero = activation_shared[thread] != T.cast(0, input_dtype)
            mask = T.ballot_sync(nonzero)
            local_position = T.popcount(
                mask & ((T.uint32(1) << lane) - T.uint32(1))
            )
            if lane == 0:
                warp_counts[warp] = T.popcount(mask)
            T.sync_threads()
            if thread == 0:
                for warp_index in T.serial(ffn_tile // 32):
                    warp_prefix[warp_index] = nonzero_count[0]
                    nonzero_count[0] += warp_counts[warp_index]
            T.sync_threads()
            if nonzero:
                nonzero_ids[warp_prefix[warp] + local_position] = thread
            T.sync_threads()
            T.clear(accumulators)
            for index in T.serial(ffn_tile):
                if index < nonzero_count[0]:
                    local_row = nonzero_ids[index]
                    actual_row = ffn_block * ffn_tile + local_row
                    activation = activation_shared[local_row]
                    for pair_lane in T.serial(2):
                        channel = (
                            channel_block * output_tile
                            + thread * 2
                            + pair_lane
                        )
                        accumulators[pair_lane] = T.cast(
                            activation * weight[actual_row, channel]
                            + accumulators[pair_lane],
                            input_dtype,
                        )
            channel = channel_block * output_tile + thread * 2
            T.atomic_addx2(
                output[channel : channel + 2], accumulators[0:2]
            )

    return kernel


@lru_cache(maxsize=16)
def _compiled_cmix_sparse_atomic(
    channels: int,
    ffn_rows: int,
    input_dtype: str,
    ffn_tile: int,
    threads: int,
    device_arch: str,
):
    del device_arch
    import tilelang  # type: ignore[import-not-found]

    return tilelang.compile(
        _build_cmix_sparse_atomic_program(
            channels,
            ffn_rows,
            input_dtype,
            ffn_tile,
            threads,
        ),
        out_idx=[],
        execution_backend="auto",
    )


def _build_cmix_sparse_binned_program(
    channels: int,
    ffn_rows: int,
    input_dtype: str = "float16",
    ffn_tile: int = 128,
    threads: int = 128,
    exponent_bins: int = 6,
):
    """Build deterministic exponent-binned exact-zero FFN accumulation."""
    output_tile = threads * 4
    if (
        channels <= 0
        or ffn_rows <= 0
        or ffn_rows % ffn_tile
        or channels % output_tile
        or exponent_bins != 6
    ):
        raise ValueError("binned FFN dimensions must divide sparse tiles")
    if input_dtype != "float16":
        raise ValueError("binned sparse FFN currently requires float16")

    import tilelang.language as T  # type: ignore[import-not-found]
    from tvm.tirx import op as tirx_op  # type: ignore[import-not-found]

    @T.prim_func
    def kernel(
        hidden: T.Tensor((ffn_rows,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        weight: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (ffn_rows, channels), input_dtype
        ),
        output: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (exponent_bins, channels), "float32"
        ),
    ):
        with T.Kernel(
            ffn_rows // ffn_tile,
            channels // output_tile,
            threads=threads,
        ) as (ffn_block, channel_block):
            thread = T.get_thread_binding(0)
            activation_shared = T.alloc_shared((ffn_tile,), input_dtype)
            nonzero_ids = T.alloc_shared((ffn_tile,), "int32")
            nonzero_count = T.alloc_shared((1,), "int32")
            warp_counts = T.alloc_shared((ffn_tile // 32,), "int32")
            warp_prefix = T.alloc_shared((ffn_tile // 32,), "int32")
            accumulators = T.alloc_local((4,), input_dtype)
            accumulator_bins = T.alloc_local((4,), "int32")
            remaining_bins = T.alloc_local((4,), "int32")
            vector_values = T.alloc_local((4,), "float32")
            ffn_row = ffn_block * ffn_tile + thread
            preactivation = T.cast(hidden[ffn_row], "float32")
            activated = T.max(preactivation, 0.0)
            activation_shared[thread] = T.cast(
                activated * activated, input_dtype
            )
            if thread == 0:
                nonzero_count[0] = 0
            T.sync_threads()
            lane = thread % 32
            warp = thread // 32
            nonzero = activation_shared[thread] != T.cast(0, input_dtype)
            mask = T.ballot_sync(nonzero)
            local_position = T.popcount(
                mask & ((T.uint32(1) << lane) - T.uint32(1))
            )
            if lane == 0:
                warp_counts[warp] = T.popcount(mask)
            T.sync_threads()
            if thread == 0:
                for warp_index in T.serial(ffn_tile // 32):
                    warp_prefix[warp_index] = nonzero_count[0]
                    nonzero_count[0] += warp_counts[warp_index]
            T.sync_threads()
            if nonzero:
                nonzero_ids[warp_prefix[warp] + local_position] = thread
            T.sync_threads()
            T.clear(accumulators)
            for index in T.serial(ffn_tile):
                if index < nonzero_count[0]:
                    local_row = nonzero_ids[index]
                    actual_row = ffn_block * ffn_tile + local_row
                    activation = activation_shared[local_row]
                    for output_lane in T.serial(4):
                        channel = (
                            channel_block * output_tile
                            + thread * 4
                            + output_lane
                        )
                        accumulators[output_lane] = T.cast(
                            activation * weight[actual_row, channel]
                            + accumulators[output_lane],
                            input_dtype,
                        )
            channel = channel_block * output_tile + thread * 4
            for output_lane in T.serial(4):
                value = accumulators[output_lane]
                bits = T.reinterpret(value, "uint16")
                exponent = T.cast(
                    T.bitwise_and(T.shift_right(bits, 10), 31),
                    "int32",
                )
                accumulator_bins[output_lane] = T.if_then_else(
                    exponent == 0,
                    0,
                    T.min(1 + (exponent - 1) // 6, exponent_bins - 1),
                )
            common_bin = T.max(
                T.max(accumulator_bins[0], accumulator_bins[1]),
                T.max(accumulator_bins[2], accumulator_bins[3]),
            )
            for output_lane in T.serial(4):
                vector_values[output_lane] = T.if_then_else(
                    accumulator_bins[output_lane] == common_bin,
                    T.cast(accumulators[output_lane], "float32"),
                    0.0,
                )
            T.call_intrin(
                "float4",
                tirx_op.Op.get("tl.atomic_addx4_elem_op"),
                T.access_ptr(
                    output[common_bin, channel : channel + 4], "rw"
                ),
                T.access_ptr(vector_values[0:4], "r"),
            )
            for output_lane in T.serial(4):
                remaining_bins[output_lane] = T.if_then_else(
                    accumulator_bins[output_lane] != common_bin
                    and accumulators[output_lane] != T.cast(0, input_dtype),
                    accumulator_bins[output_lane],
                    -1,
                )
            second_bin = T.max(
                T.max(remaining_bins[0], remaining_bins[1]),
                T.max(remaining_bins[2], remaining_bins[3]),
            )
            if second_bin >= 0:
                for output_lane in T.serial(4):
                    vector_values[output_lane] = T.if_then_else(
                        accumulator_bins[output_lane] == second_bin,
                        T.cast(accumulators[output_lane], "float32"),
                        0.0,
                    )
                T.call_intrin(
                    "float4",
                    tirx_op.Op.get("tl.atomic_addx4_elem_op"),
                    T.access_ptr(
                        output[second_bin, channel : channel + 4], "rw"
                    ),
                    T.access_ptr(vector_values[0:4], "r"),
                )
            for output_lane in T.serial(4):
                if (
                    accumulator_bins[output_lane] != common_bin
                    and accumulator_bins[output_lane] != second_bin
                    and accumulators[output_lane] != T.cast(0, input_dtype)
                ):
                    T.atomic_add(
                        output[accumulator_bins[output_lane], channel + output_lane],
                        T.cast(accumulators[output_lane], "float32"),
                    )
    return kernel


@lru_cache(maxsize=16)
def _compiled_cmix_sparse_binned(
    channels: int,
    ffn_rows: int,
    input_dtype: str,
    ffn_tile: int,
    threads: int,
    exponent_bins: int,
    device_arch: str,
):
    del device_arch
    import tilelang  # type: ignore[import-not-found]

    return tilelang.compile(
        _build_cmix_sparse_binned_program(
            channels,
            ffn_rows,
            input_dtype,
            ffn_tile,
            threads,
            exponent_bins,
        ),
        out_idx=[],
        execution_backend="auto",
    )


def _build_cmix_binned_finalize_program(
    channels: int,
    exponent_bins: int = 6,
    input_dtype: str = "float16",
    threads: int = 256,
):
    """Build fixed-order bin reduction plus FP16 residual finalization."""
    if channels <= 0 or channels % threads or exponent_bins != 6:
        raise ValueError("binned finalization dimensions are unsupported")
    if input_dtype != "float16":
        raise ValueError("binned finalization currently requires float16")

    import tilelang.language as T  # type: ignore[import-not-found]

    @T.prim_func
    def kernel(
        bins: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (exponent_bins, channels), "float32"
        ),
        residual: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        output: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
    ):
        with T.Kernel(channels // threads, threads=threads) as block:
            channel = block * threads + T.get_thread_binding(0)
            total = T.alloc_local((1,), input_dtype)
            total[0] = T.cast(bins[0, channel], input_dtype)
            bins[0, channel] = 0.0
            for exponent in T.serial(1, exponent_bins):
                total[0] = T.cast(
                    total[0] + T.cast(bins[exponent, channel], input_dtype),
                    input_dtype,
                )
                bins[exponent, channel] = 0.0
            contribution = total[0]
            output[channel] = T.cast(
                residual[channel] + contribution, input_dtype
            )

    return kernel


@lru_cache(maxsize=16)
def _compiled_cmix_binned_finalize(
    channels: int,
    exponent_bins: int,
    input_dtype: str,
    threads: int,
    device_arch: str,
):
    del device_arch
    import tilelang  # type: ignore[import-not-found]

    return tilelang.compile(
        _build_cmix_binned_finalize_program(
            channels, exponent_bins, input_dtype, threads
        ),
        out_idx=[],
        execution_backend="auto",
    )

def _build_cmix_sparse_split_program(
    channels: int,
    ffn_rows: int,
    input_dtype: str = "float16",
    ffn_tile: int = 128,
    threads: int = 128,
    splits: int = 8,
):
    """Build split exact-zero FFN down projection without global atomics."""
    output_tile = threads * 2
    if (
        channels <= 0
        or ffn_rows <= 0
        or ffn_rows % (ffn_tile * splits)
        or channels % output_tile
        or input_dtype != "float16"
    ):
        raise ValueError("FP16 sparse split dimensions must divide tiles")

    import tilelang.language as T  # type: ignore[import-not-found]


    @T.prim_func
    def kernel(
        hidden: T.Tensor((ffn_rows,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        weight: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (ffn_rows, channels), input_dtype
        ),
        partials: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (splits, channels), input_dtype
        ),
    ):
        with T.Kernel(
            splits,
            channels // output_tile,
            threads=threads,
        ) as (split, channel_block):
            thread = T.get_thread_binding(0)
            activation_shared = T.alloc_shared((ffn_tile,), input_dtype)
            nonzero_ids = T.alloc_shared((ffn_tile,), "int32")
            nonzero_count = T.alloc_shared((1,), "int32")
            warp_counts = T.alloc_shared((ffn_tile // 32,), "int32")
            warp_prefix = T.alloc_shared((ffn_tile // 32,), "int32")
            accumulators = T.alloc_local((2,), input_dtype)
            tile_accumulators = T.alloc_local((2,), input_dtype)
            T.clear(accumulators)

            for local_tile in T.serial(
                ffn_rows // (ffn_tile * splits)
            ):
                ffn_block = (
                    split * (ffn_rows // (ffn_tile * splits)) + local_tile
                )
                ffn_row = ffn_block * ffn_tile + thread
                preactivation = T.cast(hidden[ffn_row], "float32")
                activated = T.max(preactivation, 0.0)
                activation_shared[thread] = T.cast(
                    activated * activated, input_dtype
                )
                if thread == 0:
                    nonzero_count[0] = 0
                T.sync_threads()
                lane = thread % 32
                warp = thread // 32
                nonzero = activation_shared[thread] != T.cast(0, input_dtype)
                mask = T.ballot_sync(nonzero)
                local_position = T.popcount(
                    mask & ((T.uint32(1) << lane) - T.uint32(1))
                )
                if lane == 0:
                    warp_counts[warp] = T.popcount(mask)
                T.sync_threads()
                if thread == 0:
                    for warp_index in T.serial(ffn_tile // 32):
                        warp_prefix[warp_index] = nonzero_count[0]
                        nonzero_count[0] += warp_counts[warp_index]
                T.sync_threads()
                if nonzero:
                    nonzero_ids[warp_prefix[warp] + local_position] = thread
                T.sync_threads()
                T.clear(tile_accumulators)
                for index in T.serial(ffn_tile):
                    if index < nonzero_count[0]:
                        local_row = nonzero_ids[index]
                        actual_row = ffn_block * ffn_tile + local_row
                        activation = activation_shared[local_row]
                        for pair_lane in T.serial(2):
                            channel = (
                                channel_block * output_tile
                                + thread * 2
                                + pair_lane
                            )
                            tile_accumulators[pair_lane] = T.cast(
                                activation * weight[actual_row, channel]
                                + tile_accumulators[pair_lane],
                                input_dtype,
                            )
                for pair_lane in T.serial(2):
                    accumulators[pair_lane] = T.cast(
                        accumulators[pair_lane]
                        + tile_accumulators[pair_lane],
                        input_dtype,
                    )
                T.sync_threads()

            channel = channel_block * output_tile + thread * 2
            partials[split, channel] = accumulators[0]
            partials[split, channel + 1] = accumulators[1]

    return kernel


@lru_cache(maxsize=16)
def _compiled_cmix_sparse_split(
    channels: int,
    ffn_rows: int,
    input_dtype: str,
    ffn_tile: int,
    threads: int,
    splits: int,
    device_arch: str,
):
    del device_arch
    import tilelang  # type: ignore[import-not-found]

    return tilelang.compile(
        _build_cmix_sparse_split_program(
            channels,
            ffn_rows,
            input_dtype,
            ffn_tile,
            threads,
            splits,
        ),
        out_idx=[],
        execution_backend="auto",
    )

def _build_cmix_finalize_program(
    channels: int, splits: int, input_dtype: str = "float16"
):
    """Build deterministic split-output finalization."""
    if channels <= 0 or channels % 256 or splits <= 0:
        raise ValueError("channels must be positive and divisible by 256")
    if input_dtype not in {"float16", "bfloat16"}:
        raise ValueError("FFN dtype must be float16 or bfloat16")

    import tilelang.language as T  # type: ignore[import-not-found]

    @T.prim_func
    def kernel(
        partials: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (splits, channels), input_dtype
        ),
        output: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
    ):
        with T.Kernel(channels // 256, threads=256) as block:
            row = block * 256 + T.get_thread_binding(0)
            value = T.alloc_local((1,), input_dtype)
            value[0] = partials[0, row]
            for split in T.serial(1, splits):
                value[0] = T.cast(
                    value[0] + partials[split, row], input_dtype
                )
            output[row] = value[0]

    return kernel


@lru_cache(maxsize=16)
def _compiled_cmix_finalize(
    channels: int, splits: int, input_dtype: str, device_arch: str
):
    del device_arch
    import tilelang  # type: ignore[import-not-found]

    return tilelang.compile(
        _build_cmix_finalize_program(channels, splits, input_dtype),
        out_idx=[],
        execution_backend="auto",
    )


def _cmix_value_out(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    partials: torch.Tensor,
    output: torch.Tensor,
    *,
    splits: int = 4,
    block_rows: int = 64,
    reduce_threads: int = 8,
) -> None:
    """Run sparse split FFN value projection with fixed graph workspaces."""
    input_dtype = _require_contiguous_cuda(
        (hidden, weight, partials, output), "FFN value"
    )
    if hidden.dim() != 1 or weight.dim() != 2:
        raise ValueError("FFN value expects hidden [F] and weight [F,C]")
    ffn_rows, channels = weight.shape
    if (
        hidden.numel() != ffn_rows
        or ffn_rows % splits
        or tuple(partials.shape) != (splits, channels)
        or tuple(output.shape) != (channels,)
    ):
        raise ValueError("FFN value split/workspace shape mismatch")
    split_rows = ffn_rows // splits
    kernel: Any = _compiled_cmix_value(
        channels,
        split_rows,
        input_dtype,
        block_rows,
        reduce_threads,
        cuda_arch_key(hidden.device),
    )
    for split in range(splits):
        start = split * split_rows
        stop = start + split_rows
        kernel(
            hidden[start:stop],
            weight[start:stop],
            partials[split],
        )
    finalize: Any = _compiled_cmix_finalize(
        channels, splits, input_dtype, cuda_arch_key(hidden.device)
    )
    finalize(partials, output)


def _build_rankout_program(
    channels: int,
    rank_w: int,
    rank_a: int,
    rank_g: int,
    rank_v: int,
    use_value_mix: bool,
    input_dtype: str = "float16",
):
    """Build fused W/A/G/V rank-out with raw decay and pointwise gates."""
    if channels <= 0 or min(rank_w, rank_a, rank_g, rank_v) < 0:
        raise ValueError("rank-out dimensions must be non-negative")
    if channels % 256:
        raise ValueError("rank-out channels must be divisible by 256")
    if input_dtype not in {"float16", "bfloat16"}:
        raise ValueError("rank-out dtype must be float16 or bfloat16")

    import tilelang.language as T  # type: ignore[import-not-found]

    @T.prim_func
    def kernel(
        decay_rank: T.Tensor((rank_w,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        a_rank: T.Tensor((rank_a,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        g_rank: T.Tensor((rank_g,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        value_rank: T.Tensor((rank_v,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        w2: T.Tensor((rank_w, channels), input_dtype),  # type: ignore[reportInvalidTypeForm]
        a2: T.Tensor((rank_a, channels), input_dtype),  # type: ignore[reportInvalidTypeForm]
        g2: T.Tensor((rank_g, channels), input_dtype),  # type: ignore[reportInvalidTypeForm]
        v2: T.Tensor((rank_v, channels), input_dtype),  # type: ignore[reportInvalidTypeForm]
        w0: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        a0: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        v0: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        value_base: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        first_value: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        decay: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        gate_a: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        gate_g: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        value: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
    ):
        with T.Kernel(channels // 256, threads=256) as block:
            channel = block * 256 + T.get_thread_binding(0)
            decay_acc = T.alloc_local((1,), "float32")
            a_acc = T.alloc_local((1,), "float32")
            g_acc = T.alloc_local((1,), "float32")
            v_acc = T.alloc_local((1,), "float32")
            T.clear(decay_acc)
            T.clear(a_acc)
            T.clear(g_acc)
            T.clear(v_acc)
            for rank in T.serial(rank_w):
                transformed = T.cast(
                    T.tanh(T.cast(decay_rank[rank], "float32")),
                    input_dtype,
                )
                decay_acc[0] += T.cast(
                    transformed, "float32"
                ) * T.cast(w2[rank, channel], "float32")
            for rank in T.serial(rank_a):
                a_acc[0] += T.cast(
                    a_rank[rank], "float32"
                ) * T.cast(a2[rank, channel], "float32")
            for rank in T.serial(rank_g):
                transformed = T.cast(
                    1.0
                    / (
                        1.0
                        + T.exp(-T.cast(g_rank[rank], "float32"))
                    ),
                    input_dtype,
                )
                g_acc[0] += T.cast(
                    transformed, "float32"
                ) * T.cast(g2[rank, channel], "float32")
            if use_value_mix:
                for rank in T.serial(rank_v):
                    v_acc[0] += T.cast(
                        value_rank[rank], "float32"
                    ) * T.cast(v2[rank, channel], "float32")

            decay[channel] = T.cast(decay_acc[0], input_dtype)

            a_raw = T.cast(
                a0[channel] + T.cast(a_acc[0], input_dtype),
                input_dtype,
            )
            gate_a[channel] = T.cast(
                1.0 / (1.0 + T.exp(-T.cast(a_raw, "float32"))),
                input_dtype,
            )
            gate_g[channel] = T.cast(g_acc[0], input_dtype)
            if use_value_mix:
                v_raw = T.cast(
                    v0[channel] + T.cast(v_acc[0], input_dtype),
                    input_dtype,
                )
                v_gate = T.cast(
                    1.0 / (1.0 + T.exp(-T.cast(v_raw, "float32"))),
                    input_dtype,
                )
                delta = T.cast(
                    first_value[channel] - value_base[channel],
                    input_dtype,
                )
                value[channel] = T.cast(
                    value_base[channel]
                    + T.cast(delta * v_gate, input_dtype),
                    input_dtype,
                )
            else:
                value[channel] = value_base[channel]

    return kernel


def _build_rankout_reduced_program(
    channels: int,
    rank_w: int,
    rank_a: int,
    rank_g: int,
    rank_v: int,
    use_value_mix: bool,
    input_dtype: str = "float16",
    out_tile: int = 4,
    reduce_threads: int = 128,
):
    """Build rank-parallel W/A/G/V rank-out from transposed weights."""
    if channels <= 0 or channels % out_tile:
        raise ValueError("rank-out channels must be divisible by output tile")
    if min(rank_w, rank_a, rank_g, rank_v) < 0:
        raise ValueError("rank-out dimensions must be non-negative")
    if input_dtype not in {"float16", "bfloat16"}:
        raise ValueError("rank-out dtype must be float16 or bfloat16")
    if reduce_threads & (reduce_threads - 1):
        raise ValueError("rank-out reduction threads must be a power of two")

    import tilelang.language as T  # type: ignore[import-not-found]

    @T.prim_func
    def kernel(
        decay_rank: T.Tensor((rank_w,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        a_rank: T.Tensor((rank_a,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        g_rank: T.Tensor((rank_g,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        value_rank: T.Tensor((rank_v,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        w2: T.Tensor((channels, rank_w), input_dtype),  # type: ignore[reportInvalidTypeForm]
        a2: T.Tensor((channels, rank_a), input_dtype),  # type: ignore[reportInvalidTypeForm]
        g2: T.Tensor((channels, rank_g), input_dtype),  # type: ignore[reportInvalidTypeForm]
        v2: T.Tensor((channels, rank_v), input_dtype),  # type: ignore[reportInvalidTypeForm]
        a0: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        v0: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        value_base: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        first_value: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        decay: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        gate_a: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        gate_g: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        value: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
    ):
        with T.Kernel(
            channels // out_tile, threads=reduce_threads
        ) as block:
            thread = T.get_thread_binding(0)
            channel_start = block * out_tile
            accumulator = T.alloc_local((4, out_tile), "float32")
            partial = T.alloc_shared(
                (4, out_tile, reduce_threads), "float32"
            )
            T.clear(accumulator)

            for chunk in T.serial((rank_w + reduce_threads - 1) // reduce_threads):
                rank = chunk * reduce_threads + thread
                if rank < rank_w:
                    rank_value = T.tanh(T.cast(decay_rank[rank], "float32"))
                    for output_lane in T.serial(out_tile):
                        accumulator[0, output_lane] += rank_value * T.cast(
                            w2[channel_start + output_lane, rank], "float32"
                        )
            for chunk in T.serial((rank_a + reduce_threads - 1) // reduce_threads):
                rank = chunk * reduce_threads + thread
                if rank < rank_a:
                    rank_value = T.cast(a_rank[rank], "float32")
                    for output_lane in T.serial(out_tile):
                        accumulator[1, output_lane] += rank_value * T.cast(
                            a2[channel_start + output_lane, rank], "float32"
                        )
            for chunk in T.serial((rank_g + reduce_threads - 1) // reduce_threads):
                rank = chunk * reduce_threads + thread
                if rank < rank_g:
                    rank_value = 1.0 / (
                        1.0 + T.exp(-T.cast(g_rank[rank], "float32"))
                    )
                    for output_lane in T.serial(out_tile):
                        accumulator[2, output_lane] += rank_value * T.cast(
                            g2[channel_start + output_lane, rank], "float32"
                        )
            if use_value_mix:
                for chunk in T.serial(
                    (rank_v + reduce_threads - 1) // reduce_threads
                ):
                    rank = chunk * reduce_threads + thread
                    if rank < rank_v:
                        rank_value = T.cast(value_rank[rank], "float32")
                        for output_lane in T.serial(out_tile):
                            accumulator[3, output_lane] += rank_value * T.cast(
                                v2[channel_start + output_lane, rank], "float32"
                            )

            for role in T.serial(4):
                for output_lane in T.serial(out_tile):
                    partial[role, output_lane, thread] = accumulator[
                        role, output_lane
                    ]
            T.sync_threads()
            for reduction_step in T.serial(7):
                stride = reduce_threads >> (reduction_step + 1)
                if thread < stride:
                    for role in T.serial(4):
                        for output_lane in T.serial(out_tile):
                            partial[role, output_lane, thread] += partial[
                                role, output_lane, thread + stride
                            ]
                T.sync_threads()

            if thread == 0:
                for output_lane in T.serial(out_tile):
                    channel = channel_start + output_lane
                    decay[channel] = T.cast(
                        partial[0, output_lane, 0], input_dtype
                    )
                    a_raw = T.cast(
                        a0[channel]
                        + T.cast(partial[1, output_lane, 0], input_dtype),
                        input_dtype,
                    )
                    gate_a[channel] = T.cast(
                        1.0 / (1.0 + T.exp(-T.cast(a_raw, "float32"))),
                        input_dtype,
                    )
                    gate_g[channel] = T.cast(
                        partial[2, output_lane, 0], input_dtype
                    )
                    if use_value_mix:
                        v_raw = T.cast(
                            v0[channel]
                            + T.cast(partial[3, output_lane, 0], input_dtype),
                            input_dtype,
                        )
                        v_gate = T.cast(
                            1.0 / (1.0 + T.exp(-T.cast(v_raw, "float32"))),
                            input_dtype,
                        )
                        delta = T.cast(
                            first_value[channel] - value_base[channel],
                            input_dtype,
                        )
                        value[channel] = T.cast(
                            value_base[channel]
                            + T.cast(delta * v_gate, input_dtype),
                            input_dtype,
                        )
                    else:
                        value[channel] = value_base[channel]

    return kernel


@lru_cache(maxsize=32)
def _compiled_rankout_reduced(
    channels: int,
    rank_w: int,
    rank_a: int,
    rank_g: int,
    rank_v: int,
    use_value_mix: bool,
    input_dtype: str,
    out_tile: int,
    reduce_threads: int,
    device_arch: str,
):
    del device_arch
    import tilelang  # type: ignore[import-not-found]

    return tilelang.compile(
        _build_rankout_reduced_program(
            channels,
            rank_w,
            rank_a,
            rank_g,
            rank_v,
            use_value_mix,
            input_dtype,
            out_tile,
            reduce_threads,
        ),
        out_idx=[],
        execution_backend="auto",
    )

@lru_cache(maxsize=32)
def _compiled_rankout(
    channels: int,
    rank_w: int,
    rank_a: int,
    rank_g: int,
    rank_v: int,
    use_value_mix: bool,
    input_dtype: str,
    device_arch: str,
):
    del device_arch
    import tilelang  # type: ignore[import-not-found]

    return tilelang.compile(
        _build_rankout_program(
            channels,
            rank_w,
            rank_a,
            rank_g,
            rank_v,
            use_value_mix,
            input_dtype,
        ),
        out_idx=[],
        execution_backend="auto",
    )


def _rankout_out(
    ranks: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    vectors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    value_base: torch.Tensor,
    first_value: torch.Tensor,
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    use_value_mix: bool,
) -> None:
    """Run fused rank-out and pointwise finalization into fixed outputs."""
    tensors = (*ranks, *weights, *vectors, value_base, first_value, *outputs)
    input_dtype = _require_contiguous_cuda(tensors, "rank-out")
    channels = value_base.numel()
    rank_sizes = tuple(rank.numel() for rank in ranks)
    if any(tuple(rank.shape) != (size,) for rank, size in zip(ranks, rank_sizes)):
        raise ValueError("rank-out inputs must be one-dimensional")
    for weight, size in zip(weights, rank_sizes, strict=True):
        if tuple(weight.shape) != (size, channels):
            raise ValueError("rank-out weight shape mismatch")
    if any(tuple(vector.shape) != (channels,) for vector in vectors):
        raise ValueError("rank-out vectors must have shape [C]")
    if tuple(first_value.shape) != (channels,) or any(
        tuple(output.shape) != (channels,) for output in outputs
    ):
        raise ValueError("rank-out value/output shape mismatch")
    kernel: Any = _compiled_rankout(
        channels,
        *rank_sizes,
        use_value_mix,
        input_dtype,
        cuda_arch_key(value_base.device),
    )
    kernel(*ranks, *weights, *vectors, value_base, first_value, *outputs)


def _build_key_gate_program(
    num_heads: int,
    head_size: int = 64,
    input_dtype: str = "float16",
):
    """Build fused per-head key normalization and recurrent gate vectors."""
    if num_heads <= 0 or head_size != 64:
        raise ValueError("key gate requires positive heads of size 64")
    if input_dtype not in {"float16", "bfloat16"}:
        raise ValueError("key gate dtype must be float16 or bfloat16")

    import tilelang.language as T  # type: ignore[import-not-found]

    @T.prim_func
    def kernel(
        key: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (num_heads * head_size,), input_dtype
        ),
        key_scale: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (num_heads * head_size,), input_dtype
        ),
        gate_a: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (num_heads * head_size,), input_dtype
        ),
        gate_scale: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (num_heads * head_size,), input_dtype
        ),
        normalized: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (num_heads * head_size,), input_dtype
        ),
        modified: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (num_heads * head_size,), input_dtype
        ),
        anti_key: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (num_heads * head_size,), input_dtype
        ),
        anti_gate: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (num_heads * head_size,), input_dtype
        ),
    ):
        with T.Kernel(num_heads, threads=head_size) as head:
            lane = T.get_thread_binding(0)
            channel = head * head_size + lane
            local_square = T.alloc_local((1,), "float32")
            reduced_square = T.alloc_local((1,), "float32")
            scaled = T.cast(key[channel] * key_scale[channel], input_dtype)
            local_square[0] = T.cast(scaled, "float32") * T.cast(
                scaled, "float32"
            )
            with T.attr(
                T.comm_reducer(
                    lambda left, right: left + right,
                    [T.cast(0, "float32")],
                ),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        local_square[0],
                        True,
                        reduced_square[0],
                        lane,
                        dtype="handle",
                    )
                )
            norm = T.max(T.sqrt(reduced_square[0]), 1.0e-12)
            normalized_value = T.cast(
                T.cast(scaled, "float32") / norm, input_dtype
            )
            gate_delta = T.cast(gate_a[channel] - 1.0, input_dtype)
            gate_factor = T.cast(
                1.0
                + T.cast(gate_delta * gate_scale[channel], input_dtype),
                input_dtype,
            )
            modified_value = T.cast(
                key[channel] * gate_factor, input_dtype
            )
            normalized[channel] = normalized_value
            modified[channel] = modified_value
            anti_key[channel] = T.cast(-normalized_value, input_dtype)
            anti_gate[channel] = T.cast(
                normalized_value * gate_a[channel], input_dtype
            )

    return kernel


@lru_cache(maxsize=32)
def _compiled_key_gate(
    num_heads: int,
    head_size: int,
    input_dtype: str,
    device_arch: str,
):
    del device_arch
    import tilelang  # type: ignore[import-not-found]

    return tilelang.compile(
        _build_key_gate_program(num_heads, head_size, input_dtype),
        out_idx=[],
        execution_backend="auto",
    )

def _build_post_state_program(
    num_heads: int,
    head_size: int,
    input_dtype: str = "float16",
    epsilon: float = 6.4e-4,
):
    """Build fused GroupNorm, RKV residual, and gate finalization."""
    if num_heads <= 0 or head_size <= 0 or head_size > 1024:
        raise ValueError("invalid post-state dimensions")
    if input_dtype not in {"float16", "bfloat16"}:
        raise ValueError("post-state dtype must be float16 or bfloat16")
    channels = num_heads * head_size

    import tilelang.language as T  # type: ignore[import-not-found]

    @T.prim_func
    def kernel(
        projected: T.Tensor((num_heads, head_size), input_dtype),  # type: ignore[reportInvalidTypeForm]
        receptance: T.Tensor((num_heads, head_size), input_dtype),  # type: ignore[reportInvalidTypeForm]
        key: T.Tensor((num_heads, head_size), input_dtype),  # type: ignore[reportInvalidTypeForm]
        value: T.Tensor((num_heads, head_size), input_dtype),  # type: ignore[reportInvalidTypeForm]
        r_k: T.Tensor((num_heads, head_size), input_dtype),  # type: ignore[reportInvalidTypeForm]
        gate: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        norm_weight: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        norm_bias: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        output: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
    ):
        with T.Kernel(
            num_heads, threads=channels // num_heads
        ) as head:
            lane = T.get_thread_binding(0)
            channel = head * head_size + lane
            sample = T.cast(projected[head, lane], "float32")
            sample_sum = T.alloc_local((1,), "float32")
            square_sum = T.alloc_local((1,), "float32")
            rkv_sum = T.alloc_local((1,), "float32")
            reduced_sum = T.alloc_local((1,), "float32")
            reduced_square = T.alloc_local((1,), "float32")
            reduced_rkv = T.alloc_local((1,), "float32")
            sample_sum[0] = sample
            square_sum[0] = sample * sample
            rkv_sum[0] = (
                T.cast(receptance[head, lane], "float32")
                * T.cast(key[head, lane], "float32")
                * T.cast(r_k[head, lane], "float32")
            )
            with T.attr(
                T.comm_reducer(
                    lambda left, right: left + right,
                    [T.cast(0, "float32")],
                ),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        sample_sum[0],
                        True,
                        reduced_sum[0],
                        lane,
                        dtype="handle",
                    )
                )
            with T.attr(
                T.comm_reducer(
                    lambda left, right: left + right,
                    [T.cast(0, "float32")],
                ),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        square_sum[0],
                        True,
                        reduced_square[0],
                        lane,
                        dtype="handle",
                    )
                )
            with T.attr(
                T.comm_reducer(
                    lambda left, right: left + right,
                    [T.cast(0, "float32")],
                ),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        rkv_sum[0],
                        True,
                        reduced_rkv[0],
                        lane,
                        dtype="handle",
                    )
                )
            mean = reduced_sum[0] / head_size
            variance = T.max(
                reduced_square[0] / head_size - mean * mean, 0.0
            )
            normalized = (sample - mean) / T.sqrt(variance + epsilon)
            affine = (
                normalized * T.cast(norm_weight[channel], "float32")
                + T.cast(norm_bias[channel], "float32")
            )
            residual = reduced_rkv[0] * T.cast(
                value[head, lane], "float32"
            )
            output[channel] = T.cast(
                (affine + residual) * T.cast(gate[channel], "float32"),
                input_dtype,
            )

    return kernel


@lru_cache(maxsize=32)
def _compiled_post_state(
    num_heads: int,
    head_size: int,
    input_dtype: str,
    epsilon: float,
    device_arch: str,
):
    del device_arch
    import tilelang  # type: ignore[import-not-found]

    return tilelang.compile(
        _build_post_state_program(
            num_heads, head_size, input_dtype, epsilon
        ),
        out_idx=[],
        execution_backend="auto",
    )


def _post_state_out(
    projected: torch.Tensor,
    receptance: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    r_k: torch.Tensor,
    gate: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    output: torch.Tensor,
    *,
    epsilon: float,
) -> None:
    """Run fused post-state normalization and residual finalization."""
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
    input_dtype = _require_contiguous_cuda(tensors, "post-state")
    if projected.dim() != 2:
        raise ValueError("post-state projected input must have shape [H,N]")
    num_heads, head_size = projected.shape
    channels = num_heads * head_size
    if any(
        tuple(tensor.shape) != (num_heads, head_size)
        for tensor in (receptance, key, value, r_k)
    ):
        raise ValueError("post-state head tensor shape mismatch")
    if any(
        tuple(tensor.shape) != (channels,)
        for tensor in (gate, norm_weight, norm_bias, output)
    ):
        raise ValueError("post-state channel tensor shape mismatch")
    kernel: Any = _compiled_post_state(
        num_heads,
        head_size,
        input_dtype,
        epsilon,
        cuda_arch_key(projected.device),
    )
    kernel(*tensors)


def _build_rkv_program(
    channels: int,
    rank_w: int,
    rank_a: int,
    rank_g: int,
    rank_v: int,
    block_rows: int = 2,
    reduce_threads: int = 128,
    input_dtype: str = "float16",
):
    """Build output-tiled direct R/K/V plus W/A/G/V rank-input GEMV."""
    ranks = (rank_w, rank_a, rank_g, rank_v)
    if channels <= 0 or any(rank < 0 for rank in ranks):
        raise ValueError("channels must be positive and ranks non-negative")
    if input_dtype not in {"float16", "bfloat16"}:
        raise ValueError("RKV dtype must be float16 or bfloat16")
    if block_rows not in {1, 2}:
        raise ValueError("RKV output tile must contain one or two rows")
    total_rows = 3 * channels + sum(ranks)
    if any(size % block_rows for size in (channels, *ranks)):
        raise ValueError("every RKV segment must be divisible by block_rows")
    vector_width = 8
    block_k = reduce_threads * vector_width
    if channels % block_k:
        raise ValueError("channels must be divisible by reduce_threads * 8")

    import tilelang.language as T  # type: ignore[import-not-found]

    r_end = channels
    k_end = 2 * channels
    v_end = 3 * channels
    w_end = v_end + rank_w
    a_end = w_end + rank_a
    g_end = a_end + rank_g

    @T.prim_func
    def kernel(
        xr: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        xk: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        xv: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        xw: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        xa: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        xg: T.Tensor((channels,), input_dtype),  # type: ignore[reportInvalidTypeForm]
        weight: T.Tensor(  # type: ignore[reportInvalidTypeForm]
            (total_rows, channels), input_dtype
        ),
        output: T.Tensor((total_rows,), input_dtype),  # type: ignore[reportInvalidTypeForm]
    ):
        with T.Kernel(
            total_rows // block_rows, threads=reduce_threads
        ) as block:
            reduce_lane = T.get_thread_binding(0)
            row = block * block_rows
            mixed_local = T.alloc_local((vector_width,), input_dtype)
            weight_local = T.alloc_local(
                (block_rows, vector_width), input_dtype
            )
            accumulator0 = T.alloc_local((1,), "float32")
            accumulator1 = T.alloc_local((1,), "float32")
            reduced0 = T.alloc_local((1,), "float32")
            reduced1 = T.alloc_local((1,), "float32")
            T.clear(accumulator0)
            T.clear(accumulator1)
            for chunk in T.serial(channels // block_k):
                for lane in T.vectorized(vector_width):
                    column = (
                        chunk * block_k
                        + reduce_lane * vector_width
                        + lane
                    )
                    mixed_local[lane] = T.if_then_else(
                        row < r_end,
                        xr[column],
                        T.if_then_else(
                            row < k_end,
                            xk[column],
                            T.if_then_else(
                                row < v_end,
                                xv[column],
                                T.if_then_else(
                                    row < w_end,
                                    xw[column],
                                    T.if_then_else(
                                        row < a_end,
                                        xa[column],
                                        T.if_then_else(
                                            row < g_end, xg[column], xv[column]
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    )
                    weight_local[0, lane] = weight[row, column]
                    if block_rows == 2:
                        weight_local[1, lane] = weight[row + 1, column]
                for lane in T.serial(vector_width):
                    mixed_value = T.cast(mixed_local[lane], "float32")
                    accumulator0[0] += mixed_value * T.cast(
                        weight_local[0, lane], "float32"
                    )
                    if block_rows == 2:
                        accumulator1[0] += mixed_value * T.cast(
                            weight_local[1, lane], "float32"
                        )
            with T.attr(
                T.comm_reducer(
                    lambda left, right: left + right,
                    [T.cast(0, "float32")],
                ),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        accumulator0[0],
                        True,
                        reduced0[0],
                        reduce_lane,
                        dtype="handle",
                    )
                )
            if block_rows == 2:
                with T.attr(
                    T.comm_reducer(
                        lambda left, right: left + right,
                        [T.cast(0, "float32")],
                    ),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
                ):
                    T.evaluate(
                        T.tvm_thread_allreduce(
                            T.uint32(1),
                            accumulator1[0],
                            True,
                            reduced1[0],
                            reduce_lane,
                            dtype="handle",
                        )
                    )
            if reduce_lane == 0:
                output[row] = T.cast(reduced0[0], input_dtype)
                if block_rows == 2:
                    output[row + 1] = T.cast(reduced1[0], input_dtype)

    return kernel


@lru_cache(maxsize=32)
def _compiled_rkv(
    channels: int,
    rank_w: int,
    rank_a: int,
    rank_g: int,
    rank_v: int,
    block_rows: int,
    reduce_threads: int,
    input_dtype: str,
    device_arch: str,
):
    del device_arch
    import tilelang  # type: ignore[import-not-found]

    return tilelang.compile(
        _build_rkv_program(
            channels,
            rank_w,
            rank_a,
            rank_g,
            rank_v,
            block_rows,
            reduce_threads,
            input_dtype,
        ),
        out_idx=[],
        execution_backend="auto",
    )


def _rkv_out(
    inputs: tuple[torch.Tensor, ...],
    weight: torch.Tensor,
    output: torch.Tensor,
    ranks: tuple[int, int, int, int],
    *,
    block_rows: int = 1,
    reduce_threads: int = 128,
) -> None:
    """Run direct-input packed RKV and rank-input projections."""
    if len(inputs) != 6:
        raise ValueError("RKV expects xr, xk, xv, xw, xa, and xg")
    input_dtype = _require_contiguous_cuda(
        (*inputs, weight, output), "RKV"
    )
    if len(ranks) != 4 or any(rank < 0 for rank in ranks):
        raise ValueError("ranks must contain four non-negative values")
    channels = inputs[0].numel()
    if any(tuple(value.shape) != (channels,) for value in inputs):
        raise ValueError("every RKV input must have shape [C]")
    total_rows = 3 * channels + sum(ranks)
    if tuple(weight.shape) != (total_rows, channels):
        raise ValueError("packed RKV low-rank weight shape mismatch")
    if tuple(output.shape) != (total_rows,):
        raise ValueError("packed RKV low-rank output shape mismatch")
    kernel: Any = _compiled_rkv(
        channels,
        *ranks,
        block_rows,
        reduce_threads,
        input_dtype,
        cuda_arch_key(inputs[0].device),
    )
    kernel(*inputs, weight, output)


def clear_tilelang_kernel_caches() -> None:
    """Drop bounded Python references to compiled decode kernels."""
    for compiler in (
        _compiled_wkv,
        _compiled_wkv_w0_t1,
        _compiled_gemv,
        _compiled_ffn,
        _compiled_tmix_layernorm_mix6,
        _compiled_cmix_add_layernorm_mix,
        _compiled_cmix_layernorm_mix,
        _compiled_cmix_value,
        _compiled_cmix_sparse_atomic,
        _compiled_cmix_sparse_binned,
        _compiled_cmix_binned_finalize,
        _compiled_cmix_sparse_split,
        _compiled_cmix_finalize,
        _compiled_rankout_reduced,
        _compiled_rankout,
        _compiled_key_gate,
        _compiled_post_state,
        _compiled_rkv,
    ):
        compiler.cache_clear()
