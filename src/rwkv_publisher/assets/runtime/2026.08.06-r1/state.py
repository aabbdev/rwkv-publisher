from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from typing import Callable, Iterator, cast
import torch
from transformers.cache_utils import Cache

TORCH_ZEROS = torch.zeros  # type: ignore[attr-defined]
FLOAT32 = torch.float32  # type: ignore[attr-defined]
INT32 = torch.int32  # type: ignore[attr-defined]


@dataclass
class RWKV7LayerState:
    channel: torch.Tensor
    time_shift: torch.Tensor
    time_matrix: torch.Tensor

    def detach(self) -> "RWKV7LayerState":
        return RWKV7LayerState(
            self.channel.detach(),
            self.time_shift.detach(),
            self.time_matrix.detach(),
        )


class RWKV7State(Cache):
    """Explicit recurrent state used as Hugging Face generation cache."""

    def __init__(
        self,
        layer_states: list[RWKV7LayerState],
        seen_tokens: int = 0,
        elapsed_tokens: torch.Tensor | None = None,
    ) -> None:
        super().__init__(layers=[])
        self.layer_states = layer_states
        self.seen_tokens = seen_tokens
        if elapsed_tokens is None and layer_states:
            reference = layer_states[0].channel
            elapsed_tokens = torch.full(
                (reference.shape[0],),
                seen_tokens,
                device=reference.device,
                dtype=INT32,
            )
        self.elapsed_tokens = elapsed_tokens

    @classmethod
    def empty(
        cls,
        *,
        num_layers: int,
        batch_size: int,
        hidden_size: int,
        num_heads: int,
        head_size: int,
        device,
        dtype,
        matrix_dtype=None,
    ) -> "RWKV7State":
        layers = [
            RWKV7LayerState(
                channel=TORCH_ZEROS(
                    batch_size, hidden_size, device=device, dtype=dtype
                ),
                time_shift=TORCH_ZEROS(
                    batch_size, hidden_size, device=device, dtype=dtype
                ),
                time_matrix=TORCH_ZEROS(
                    batch_size,
                    num_heads,
                    head_size,
                    head_size,
                    device=device,
                    dtype=matrix_dtype or FLOAT32,
                ),
            )
            for _ in range(num_layers)
        ]
        return cls(
            layers,
            elapsed_tokens=TORCH_ZEROS(
                batch_size, device=device, dtype=INT32
            ),
        )

    @property
    def is_compileable(self) -> bool:
        return False

    @property
    def is_initialized(self) -> bool:
        return bool(self.layer_states)

    @property
    def max_batch_size(self) -> int:
        if not self.layer_states:
            return 0
        return self.layer_states[0].channel.shape[0]

    @property
    def max_cache_len(self) -> int:
        return self.seen_tokens

    def get_seq_length(self, layer_idx: int = 0) -> int:
        del layer_idx
        return self.seen_tokens

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        del layer_idx
        return self.seen_tokens

    def detach(self) -> "RWKV7State":
        return RWKV7State(
            [layer.detach() for layer in self.layer_states],
            self.seen_tokens,
            None if self.elapsed_tokens is None else self.elapsed_tokens.detach(),
        )

    def clone(self) -> "RWKV7State":
        return RWKV7State(
            [
                RWKV7LayerState(
                    layer.channel.clone(),
                    layer.time_shift.clone(),
                    layer.time_matrix.clone(),
                )
                for layer in self.layer_states
            ],
            self.seen_tokens,
            None if self.elapsed_tokens is None else self.elapsed_tokens.clone(),
        )

    def copy_(self, source: "RWKV7State") -> "RWKV7State":
        """Copy another same-layout state without changing destination addresses."""
        if len(self.layer_states) != len(source.layer_states):
            raise ValueError("RWKV states have different layer counts")
        for destination, current in zip(
            self.layer_states, source.layer_states, strict=True
):
            for target, value in (
                (destination.channel, current.channel),
                (destination.time_shift, current.time_shift),
                (destination.time_matrix, current.time_matrix),
            ):
                if (
                    target.shape != value.shape
                    or target.dtype != value.dtype
                    or target.device != value.device
                ):
                    raise ValueError("RWKV states have different tensor layouts")
                target.copy_(value)
        if (self.elapsed_tokens is None) != (source.elapsed_tokens is None):
            raise ValueError("RWKV states disagree on elapsed-token storage")
        if self.elapsed_tokens is not None and source.elapsed_tokens is not None:
            if (
                self.elapsed_tokens.shape != source.elapsed_tokens.shape
                or self.elapsed_tokens.device != source.elapsed_tokens.device
            ):
                raise ValueError("RWKV elapsed-token layouts differ")
            self.elapsed_tokens.copy_(source.elapsed_tokens)
        self.seen_tokens = source.seen_tokens
        return self

    def copy_batch_row_(
        self, source: "RWKV7State", index: int, *, seen_tokens: int
) -> "RWKV7State":
        """Copy one row from a batched state into this stable one-request state."""
        if self.max_batch_size != 1:
            raise ValueError("destination state must contain exactly one request")
        if index < 0 or index >= source.max_batch_size:
            raise IndexError("source state row is out of range")
        if len(self.layer_states) != len(source.layer_states):
            raise ValueError("RWKV states have different layer counts")
        for destination, current in zip(
            self.layer_states, source.layer_states, strict=True
):
            for target, value in (
                (destination.channel, current.channel[index : index + 1]),
                (destination.time_shift, current.time_shift[index : index + 1]),
                (destination.time_matrix, current.time_matrix[index : index + 1]),
            ):
                if (
                    target.shape != value.shape
                    or target.dtype != value.dtype
                    or target.device != value.device
                ):
                    raise ValueError("RWKV states have different tensor layouts")
                target.copy_(value)
        if (self.elapsed_tokens is None) != (source.elapsed_tokens is None):
            raise ValueError("RWKV states disagree on elapsed-token storage")
        if self.elapsed_tokens is not None and source.elapsed_tokens is not None:
            self.elapsed_tokens.copy_(source.elapsed_tokens[index : index + 1])
        self.seen_tokens = seen_tokens
        return self

    @classmethod
    def batch_stack(cls, states: list["RWKV7State"]) -> "RWKV7State":
        """Stack independent request states into one decode batch."""
        if not states:
            raise ValueError("at least one RWKV state is required")
        layer_count = len(states[0].layer_states)
        elapsed_present = states[0].elapsed_tokens is not None
        if any(len(state.layer_states) != layer_count for state in states):
            raise ValueError("RWKV states have different layer counts")
        if any(
            (state.elapsed_tokens is not None) != elapsed_present for state in states
):
            raise ValueError("RWKV states disagree on elapsed-token storage")
        layers: list[RWKV7LayerState] = []
        for layer_index in range(layer_count):
            current = [state.layer_states[layer_index] for state in states]
            reference = current[0]
            for layer in current[1:]:
                for value, expected in (
                    (layer.channel, reference.channel),
                    (layer.time_shift, reference.time_shift),
                    (layer.time_matrix, reference.time_matrix),
                ):
                    if (
                        value.shape[1:] != expected.shape[1:]
                        or value.dtype != expected.dtype
                        or value.device != expected.device
                    ):
                        raise ValueError("RWKV states have incompatible tensor layouts")
            layers.append(
                RWKV7LayerState(
                    torch.cat([layer.channel for layer in current], dim=0),
                    torch.cat([layer.time_shift for layer in current], dim=0),
                    torch.cat([layer.time_matrix for layer in current], dim=0),
                )
            )
        elapsed = (
            torch.cat(
                [cast(torch.Tensor, state.elapsed_tokens) for state in states],
                dim=0,
            )
            if elapsed_present
            else None
        )
        return cls(layers, max(state.seen_tokens for state in states), elapsed)

    def batch_split(
        self,
        batch_sizes: list[int] | None = None,
        *,
        seen_tokens: list[int] | None = None,
    ) -> list["RWKV7State"]:
        """Split a batched state into cloned independent request states."""
        if batch_sizes is None:
            batch_sizes = [1] * self.max_batch_size
        if not batch_sizes or any(size < 1 for size in batch_sizes):
            raise ValueError("batch_sizes must contain positive values")
        if sum(batch_sizes) != self.max_batch_size:
            raise ValueError("batch_sizes do not cover the RWKV state batch")
        if seen_tokens is not None and len(seen_tokens) != len(batch_sizes):
            raise ValueError("seen_tokens must match batch_sizes")
        outputs: list[RWKV7State] = []
        start = 0
        for index, size in enumerate(batch_sizes):
            stop = start + size
            outputs.append(
                RWKV7State(
                    [
                        RWKV7LayerState(
                            layer.channel[start:stop].clone(),
                            layer.time_shift[start:stop].clone(),
                            layer.time_matrix[start:stop].clone(),
                        )
                        for layer in self.layer_states
                    ],
                    self.seen_tokens
                    if seen_tokens is None
                    else seen_tokens[index],
                    None
                    if self.elapsed_tokens is None
                    else self.elapsed_tokens[start:stop].clone(),
                )
            )
            start = stop
        return outputs

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self.batch_select_indices(beam_idx)

    def batch_repeat_interleave(self, repeats: int) -> None:
        for layer in self.layer_states:
            layer.channel = layer.channel.repeat_interleave(repeats, dim=0)
            layer.time_shift = layer.time_shift.repeat_interleave(repeats, dim=0)
            layer.time_matrix = layer.time_matrix.repeat_interleave(repeats, dim=0)
        if self.elapsed_tokens is not None:
            self.elapsed_tokens = self.elapsed_tokens.repeat_interleave(repeats, dim=0)
    def batch_select_indices(self, indices: torch.Tensor) -> None:
        for layer in self.layer_states:
            device_indices = indices.to(layer.channel.device)
            layer.channel = layer.channel.index_select(0, device_indices)
            layer.time_shift = layer.time_shift.index_select(0, device_indices)
            layer.time_matrix = layer.time_matrix.index_select(0, device_indices)
        if self.elapsed_tokens is not None:
            device_indices = indices.to(self.elapsed_tokens.device)
            self.elapsed_tokens = self.elapsed_tokens.index_select(0, device_indices)
    def crop(self, max_length: int) -> None:
        if max_length < self.seen_tokens:
            raise NotImplementedError(
                "RWKV recurrent state cannot roll back without state history"
            )

    def reset(self) -> None:
        for layer in self.layer_states:
            layer.channel.zero_()
            layer.time_shift.zero_()
            layer.time_matrix.zero_()
        if self.elapsed_tokens is not None:
            self.elapsed_tokens.zero_()
        self.seen_tokens = 0


class RWKV7StatePool:
    """Bounded pool that zeroes recurrent state between requests."""

    def __init__(
        self, factory: Callable[[], RWKV7State], *, max_entries: int = 8
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.factory = factory
        self.max_entries = max_entries
        prototype = factory()
        with torch.inference_mode():
            prototype.reset()
        self._signature = self._state_signature(prototype)
        self._available = [prototype]
        self._leased: dict[int, RWKV7State] = {}
        self._total = 1
        self._closed = False
        self._lock = Lock()

    @staticmethod
    def _state_signature(state: RWKV7State) -> tuple:
        tensors = [
            tensor
            for layer in state.layer_states
            for tensor in (layer.channel, layer.time_shift, layer.time_matrix)
        ]
        if state.elapsed_tokens is not None:
            tensors.append(state.elapsed_tokens)
        return (
            len(state.layer_states),
            state.elapsed_tokens is not None,
            tuple(
                (tuple(tensor.shape), tensor.dtype, str(tensor.device))
                for tensor in tensors
            ),
        )

    def acquire(self) -> RWKV7State:
        with self._lock:
            if self._closed:
                raise RuntimeError("RWKV state pool is closed")
            if self._available:
                state = self._available.pop()
            elif self._total < self.max_entries:
                state = self.factory()
                if self._state_signature(state) != self._signature:
                    raise RuntimeError("RWKV state factory changed tensor layout")
                self._total += 1
            else:
                raise RuntimeError("RWKV state pool is exhausted")
            with torch.inference_mode():
                state.reset()
            self._leased[id(state)] = state
            return state

    def release(self, state: RWKV7State) -> None:
        with self._lock:
            leased = self._leased.pop(id(state), None)
            if leased is not state:
                raise ValueError("state is not leased from this pool")
            if self._state_signature(state) != self._signature:
                self._total -= 1
                raise ValueError("state tensor layout changed while leased")
            with torch.inference_mode():
                state.reset()
            if self._closed:
                self._total -= 1
            else:
                self._available.append(state)

    def gather(self, states: list[RWKV7State]) -> RWKV7State:
        """Gather leased request states into a contiguous decode batch."""
        with self._lock:
            if any(self._leased.get(id(state)) is not state for state in states):
                raise ValueError("all states must be leased from this pool")
            return type(states[0]).batch_stack(states)

    def scatter_(
        self,
        source: RWKV7State,
        states: list[RWKV7State],
        *,
        seen_tokens: list[int],
    ) -> None:
        """Scatter a decode batch directly into stable leased destinations."""
        if len(states) != source.max_batch_size or len(seen_tokens) != len(states):
            raise ValueError("scatter rows must match leased states")
        with self._lock, torch.inference_mode():
            if any(self._leased.get(id(state)) is not state for state in states):
                raise ValueError("all states must be leased from this pool")
            for index, (state, seen) in enumerate(zip(states, seen_tokens, strict=True)):
                state.copy_batch_row_(source, index, seen_tokens=seen)
    @contextmanager
    def lease(self) -> Iterator[RWKV7State]:
        state = self.acquire()
        try:
            yield state
        finally:
            self.release(state)

    def clear(self) -> None:
        with self._lock:
            self._total -= len(self._available)
            self._available.clear()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._total -= len(self._available)
            self._available.clear()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "max_entries": self.max_entries,
                "total": self._total,
                "available": len(self._available),
                "leased": len(self._leased),
            }


@dataclass
class _RWKV7StatePage:
    page_id: int
    state: RWKV7State
    free_slots: list[int]
    leased_slots: set[int]
    allocated_bytes: int

    @property
    def slots(self) -> int:
        return self.state.max_batch_size


class RWKV7PagedStatePool:
    """Bounded lazy page allocator for constant-size RWKV recurrent states."""

    def __init__(
        self,
        factory: Callable[[], RWKV7State],
        *,
        max_entries: int = 8,
        page_size: int = 8,
        contiguous_views: bool = True,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        if page_size < 1:
            raise ValueError("page_size must be positive")
        prototype = factory()
        if prototype.max_batch_size != 1:
            raise ValueError("paged state factory must create one-request states")
        with torch.inference_mode():
            prototype.reset()
        self.factory = factory
        self.max_entries = max_entries
        self.page_size = min(page_size, max_entries)
        self.contiguous_views = contiguous_views
        self.max_pages = (max_entries + self.page_size - 1) // self.page_size
        self._signature = RWKV7StatePool._state_signature(prototype)
        self._state_type = type(prototype)
        self._layer_type = type(prototype.layer_states[0]) if prototype.layer_states else RWKV7LayerState
        self._layer_specs = [
            tuple(
                (tuple(tensor.shape[1:]), tensor.dtype, tensor.device)
                for tensor in (layer.channel, layer.time_shift, layer.time_matrix)
            )
            for layer in prototype.layer_states
        ]
        self._elapsed_spec = (
            None
            if prototype.elapsed_tokens is None
            else (prototype.elapsed_tokens.dtype, prototype.elapsed_tokens.device)
        )
        self._pages: dict[int, _RWKV7StatePage] = {}
        self._leased: dict[int, tuple[RWKV7State, int, int]] = {}
        self._next_page_id = 0
        self._closed = False
        self._contiguous_view_batches = 0
        self._gather_copy_batches = 0
        self._gathered_rows = 0
        self._lock = Lock()

    @staticmethod
    def _state_bytes(state: RWKV7State) -> int:
        tensors = [
            tensor
            for layer in state.layer_states
            for tensor in (layer.channel, layer.time_shift, layer.time_matrix)
        ]
        if state.elapsed_tokens is not None:
            tensors.append(state.elapsed_tokens)
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def _allocate_page(self) -> _RWKV7StatePage:
        allocated_slots = sum(page.slots for page in self._pages.values())
        slots = min(self.page_size, self.max_entries - allocated_slots)
        if slots < 1:
            raise RuntimeError("RWKV paged state pool is exhausted")
        layers = []
        for specs in self._layer_specs:
            tensors = [
                torch.zeros((slots, *shape), dtype=dtype, device=device)
                for shape, dtype, device in specs
            ]
            layers.append(self._layer_type(*tensors))
        elapsed = (
            None
            if self._elapsed_spec is None
            else torch.zeros(
                slots, dtype=self._elapsed_spec[0], device=self._elapsed_spec[1]
            )
        )
        page_state = self._state_type(layers, 0, elapsed)
        page = _RWKV7StatePage(
            page_id=self._next_page_id,
            state=page_state,
            free_slots=list(range(slots - 1, -1, -1)),
            leased_slots=set(),
            allocated_bytes=self._state_bytes(page_state),
        )
        self._next_page_id += 1
        self._pages[page.page_id] = page
        return page

    def _slot_view(self, page: _RWKV7StatePage, slot: int) -> RWKV7State:
        layers = [
            self._layer_type(
                layer.channel[slot : slot + 1],
                layer.time_shift[slot : slot + 1],
                layer.time_matrix[slot : slot + 1],
            )
            for layer in page.state.layer_states
        ]
        elapsed = (
            None
            if page.state.elapsed_tokens is None
            else page.state.elapsed_tokens[slot : slot + 1]
        )
        return self._state_type(layers, 0, elapsed)

    def acquire(self) -> RWKV7State:
        with self._lock:
            if self._closed:
                raise RuntimeError("RWKV paged state pool is closed")
            page = next(
                (current for current in self._pages.values() if current.free_slots),
                None,
            )
            if page is None:
                page = self._allocate_page()
            slot = page.free_slots.pop()
            page.leased_slots.add(slot)
            state = self._slot_view(page, slot)
            with torch.inference_mode():
                state.reset()
            self._leased[id(state)] = (state, page.page_id, slot)
            return state

    def release(self, state: RWKV7State) -> None:
        with self._lock:
            lease = self._leased.pop(id(state), None)
            if lease is None or lease[0] is not state:
                raise ValueError("state is not leased from this paged pool")
            _, page_id, slot = lease
            page = self._pages[page_id]
            layout_valid = RWKV7StatePool._state_signature(state) == self._signature
            stored = self._slot_view(page, slot)
            with torch.inference_mode():
                stored.reset()
            page.leased_slots.remove(slot)
            if self._closed and not page.leased_slots:
                del self._pages[page_id]
            else:
                page.free_slots.append(slot)
            if not layout_valid:
                raise ValueError("state tensor layout changed while leased")

    @contextmanager
    def lease(self) -> Iterator[RWKV7State]:
        state = self.acquire()
        try:
            yield state
        finally:
            self.release(state)

    def _contiguous_batch_view_locked(
        self, states: list[RWKV7State]
    ) -> RWKV7State | None:
        leases = [self._leased[id(state)] for state in states]
        page_id = leases[0][1]
        if any(lease[1] != page_id for lease in leases):
            return None
        slots = [lease[2] for lease in leases]
        start = slots[0]
        if slots != list(range(start, start + len(slots))):
            return None
        page = self._pages[page_id]
        stop = start + len(slots)
        layers = [
            self._layer_type(
                layer.channel[start:stop],
                layer.time_shift[start:stop],
                layer.time_matrix[start:stop],
            )
            for layer in page.state.layer_states
        ]
        elapsed = (
            None
            if page.state.elapsed_tokens is None
            else page.state.elapsed_tokens[start:stop]
        )
        return self._state_type(
            layers, max(state.seen_tokens for state in states), elapsed
        )

    def gather(self, states: list[RWKV7State]) -> RWKV7State:
        """Return a zero-copy contiguous page view or a copied fallback batch."""
        if not states:
            raise ValueError("at least one leased state is required")
        with self._lock:
            if any(
                (lease := self._leased.get(id(state))) is None
                or lease[0] is not state
                for state in states
            ):
                raise ValueError("all states must be leased from this paged pool")
            self._gathered_rows += len(states)
            view = (
                self._contiguous_batch_view_locked(states)
                if self.contiguous_views
                else None
            )
            if view is not None:
                self._contiguous_view_batches += 1
                return view
            self._gather_copy_batches += 1
            return type(states[0]).batch_stack(states)

    def scatter_(
        self,
        source: RWKV7State,
        states: list[RWKV7State],
        *,
        seen_tokens: list[int],
    ) -> None:
        if len(states) != source.max_batch_size or len(seen_tokens) != len(states):
            raise ValueError("scatter rows must match leased states")
        with self._lock, torch.inference_mode():
            if any(
                (lease := self._leased.get(id(state))) is None or lease[0] is not state
                for state in states
            ):
                raise ValueError("all states must be leased from this paged pool")
            for index, (state, seen) in enumerate(zip(states, seen_tokens, strict=True)):
                state.copy_batch_row_(source, index, seen_tokens=seen)

    def lease_info(self, state: RWKV7State) -> dict[str, int]:
        with self._lock:
            lease = self._leased.get(id(state))
            if lease is None or lease[0] is not state:
                raise ValueError("state is not leased from this paged pool")
            return {"page_id": lease[1], "slot": lease[2]}

    def clear(self) -> None:
        with self._lock:
            empty_pages = [
                page_id
                for page_id, page in self._pages.items()
                if not page.leased_slots
            ]
            for page_id in empty_pages:
                del self._pages[page_id]

    def close(self) -> None:
        with self._lock:
            self._closed = True
            empty_pages = [
                page_id
                for page_id, page in self._pages.items()
                if not page.leased_slots
            ]
            for page_id in empty_pages:
                del self._pages[page_id]

    def stats(self) -> dict[str, int | str]:
        with self._lock:
            return {
                "kind": "paged",
                "max_entries": self.max_entries,
                "page_size": self.page_size,
                "contiguous_views": int(self.contiguous_views),
                "max_pages": self.max_pages,
                "pages": len(self._pages),
                "total": sum(page.slots for page in self._pages.values()),
                "available": sum(len(page.free_slots) for page in self._pages.values()),
                "leased": len(self._leased),
                "contiguous_view_batches": self._contiguous_view_batches,
                "gather_copy_batches": self._gather_copy_batches,
                "gathered_rows": self._gathered_rows,
                "allocated_bytes": sum(
                    page.allocated_bytes for page in self._pages.values()
                ),
            }
