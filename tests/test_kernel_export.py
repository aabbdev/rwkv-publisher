from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from rwkv_publisher.runtime_export import build_kernel_export

STATE_SOURCE = """_collision = "state"

def cuda_arch_key():
    return "sm_test"

def collision_value():
    return _collision

def annotation_value(input_dtype):
    def kernel(value: input_dtype):
        return value
    return kernel.__annotations__["value"]

def inspected_source():
    import inspect
    return inspect.getsource(collision_value)
"""

DECODE_SOURCE = """from .kernel_tilelang_state import cuda_arch_key

_collision = "decode"

def collision_value():
    return _collision

def architecture():
    return cuda_arch_key()
"""


def _runtime(tmp_path: Path) -> Path:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    runtime.joinpath("kernel_tilelang_state.py").write_text(
        STATE_SOURCE, encoding="utf-8"
    )
    runtime.joinpath("kernel_tilelang_decode.py").write_text(
        DECODE_SOURCE, encoding="utf-8"
    )
    return runtime


def test_kernel_export_is_deterministic_and_isolates_collisions(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    first = build_kernel_export(runtime)
    second = build_kernel_export(runtime)

    assert first == second
    assert (
        hashlib.sha256(first.kernel.encode()).hexdigest()
        == first.provenance["output_sha256"]
    )
    namespace: dict[str, object] = {}
    exec(  # noqa: S102
        compile(first.kernel, "kernel.py", "exec", dont_inherit=True), namespace
    )
    state = namespace["state"]
    decode = namespace["decode"]
    assert state.collision_value() == "state"  # type: ignore[attr-defined]
    assert decode.collision_value() == "decode"  # type: ignore[attr-defined]
    assert decode.architecture() == "sm_test"  # type: ignore[attr-defined]
    assert state._collision != decode._collision  # type: ignore[attr-defined]
    assert state.annotation_value("float16") == "float16"  # type: ignore[attr-defined]
    assert "def collision_value():" in state.inspected_source()  # type: ignore[attr-defined]


def test_kernel_export_rejects_global_namespace_mutation(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.joinpath("kernel_tilelang_state.py").write_text(
        "value = 1\n\ndef mutate():\n    global value\n    value = 2\n",
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="unsupported namespace mutation"):
        build_kernel_export(runtime)
