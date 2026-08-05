from __future__ import annotations

from typing import Any

import torch

FLOAT32 = torch.float32  # type: ignore[attr-defined]


def _validate_state_finalize_inputs(
    state: torch.Tensor,
    decay: torch.Tensor,
    anti_update: torch.Tensor,
    value_key: torch.Tensor,
) -> None:
    if state.ndim != 4:
        raise ValueError("state must have shape [batch, heads, head_size, head_size]")
    batch_size, num_heads, rows, columns = state.shape
    if rows != columns:
        raise ValueError("state matrices must be square")
    if decay.shape != (batch_size, num_heads, columns):
        raise ValueError("decay must have shape [batch, heads, head_size]")
    if anti_update.shape != state.shape:
        raise ValueError("anti_update must match state shape")
    if value_key.shape != state.shape:
        raise ValueError("value_key must match state shape")
    tensors = (state, decay, anti_update, value_key)
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise ValueError("state finalize custom op requires CUDA tensors")
    if any(tensor.device != state.device for tensor in tensors[1:]):
        raise ValueError("state finalize tensors must use one CUDA device")
    if state.dtype != FLOAT32:
        raise ValueError("state must use float32")
    if anti_update.dtype != FLOAT32 or value_key.dtype != FLOAT32:
        raise ValueError("anti_update and value_key must use float32")
    supported_decay_dtypes = {
        FLOAT32,
        getattr(torch, "float16"),
        getattr(torch, "bfloat16"),
    }
    if decay.dtype not in supported_decay_dtypes:
        raise ValueError("decay must use float32, float16, or bfloat16")


def _validate_state_finalize_backward_inputs(
    grad_output: torch.Tensor, decay: torch.Tensor
) -> None:
    if grad_output.ndim != 4 or grad_output.dtype != FLOAT32:
        raise ValueError("grad_output must be a rank-4 float32 tensor")
    batch_size, num_heads, rows, columns = grad_output.shape
    if rows != columns or decay.shape != (batch_size, num_heads, columns):
        raise ValueError("state finalize backward shapes are incompatible")
    if grad_output.device != decay.device:
        raise ValueError("state finalize backward tensors must use one CUDA device")
    if not grad_output.is_cuda or not decay.is_cuda:
        raise ValueError("state finalize backward custom op requires CUDA tensors")


def _register_state_finalize_backward_op() -> Any:
    namespace = torch.ops.rwkv7_pytorch
    if hasattr(namespace, "_state_finalize_backward"):
        return namespace._state_finalize_backward.default

    @torch.library.custom_op(
        "rwkv7_pytorch::_state_finalize_backward",
        mutates_args=(),
        device_types="cuda",
    )
    def state_finalize_backward(
        grad_output: torch.Tensor,
        decay: torch.Tensor,
    ) -> torch.Tensor:
        _validate_state_finalize_backward_inputs(grad_output, decay)
        from .kernel_tilelang_state import tilelang_state_finalize_backward

        return tilelang_state_finalize_backward(grad_output, decay)

    @state_finalize_backward.register_fake
    def state_finalize_backward_fake(
        grad_output: torch.Tensor, decay: torch.Tensor
    ) -> torch.Tensor:
        del decay
        return grad_output.new_empty(grad_output.shape)

    return state_finalize_backward


_STATE_FINALIZE_BACKWARD_OP = _register_state_finalize_backward_op()


def _register_state_finalize_op() -> Any:
    namespace = torch.ops.rwkv7_pytorch
    if hasattr(namespace, "state_finalize"):
        return namespace.state_finalize.default

    @torch.library.custom_op(
        "rwkv7_pytorch::state_finalize",
        mutates_args=(),
        device_types="cuda",
    )
    def state_finalize(
        state: torch.Tensor,
        decay: torch.Tensor,
        anti_update: torch.Tensor,
        value_key: torch.Tensor,
    ) -> torch.Tensor:
        _validate_state_finalize_inputs(state, decay, anti_update, value_key)
        from .kernel_tilelang_state import tilelang_state_finalize

        return tilelang_state_finalize(state, decay, anti_update, value_key)

    @state_finalize.register_fake
    def state_finalize_fake(
        state: torch.Tensor,
        decay: torch.Tensor,
        anti_update: torch.Tensor,
        value_key: torch.Tensor,
    ) -> torch.Tensor:
        del decay, anti_update, value_key
        return state.new_empty(state.shape)

    def setup_context(ctx: Any, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        del output
        state, decay, _, _ = inputs
        ctx.save_for_backward(state, decay)

    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state, decay = ctx.saved_tensors
        if torch.is_grad_enabled():
            grad_state = grad_output * decay.float().unsqueeze(-2)
        else:
            grad_state = _STATE_FINALIZE_BACKWARD_OP(grad_output, decay)
        grad_decay = (grad_output * state).sum(dim=-2).to(decay.dtype)
        return grad_state, grad_decay, grad_output, grad_output

    state_finalize.register_autograd(backward, setup_context=setup_context)
    return state_finalize


_STATE_FINALIZE_OP = _register_state_finalize_op()


def tilelang_state_finalize_op(
    state: torch.Tensor,
    decay: torch.Tensor,
    anti_update: torch.Tensor,
    value_key: torch.Tensor,
) -> torch.Tensor:
    """Run differentiable TileLang state finalization through PyTorch dispatcher."""
    _validate_state_finalize_inputs(state, decay, anti_update, value_key)
    return _STATE_FINALIZE_OP(state, decay, anti_update, value_key)
