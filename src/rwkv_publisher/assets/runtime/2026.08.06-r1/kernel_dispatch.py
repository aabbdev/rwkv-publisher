from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from importlib.util import find_spec
from typing import Any

import torch

SUPPORTED_TILELANG_VERSIONS = ("0.1.12",)


@dataclass(frozen=True)
class KernelBackendStatus:
    name: str
    available: bool
    version: str | None = None
    reason: str | None = None


@lru_cache(maxsize=1)
def tilelang_status() -> KernelBackendStatus:
    if find_spec("tilelang") is None:
        return KernelBackendStatus("tilelang", False, reason="package not installed")
    installed_version = None
    with suppress(PackageNotFoundError):
        installed_version = distribution_version("tilelang")
    if installed_version is None:
        return KernelBackendStatus(
            "tilelang", False, reason="distribution metadata unavailable"
        )
    if installed_version not in SUPPORTED_TILELANG_VERSIONS:
        supported = ", ".join(SUPPORTED_TILELANG_VERSIONS)
        return KernelBackendStatus(
            "tilelang",
            False,
            version=installed_version,
            reason=f"unsupported version; expected one of: {supported}",
        )
    if not torch.cuda.is_available():
        return KernelBackendStatus(
            "tilelang", False, version=installed_version, reason="CUDA unavailable"
        )
    return KernelBackendStatus("tilelang", True, version=installed_version)


def kernel_backend_status() -> dict[str, KernelBackendStatus]:
    return {
        "torch": KernelBackendStatus("torch", True, version=torch.__version__),
        "tilelang": tilelang_status(),
    }


def available_backends() -> tuple[str, ...]:
    return tuple(
        name for name, status in kernel_backend_status().items() if status.available
    )


def resolve_backend(requested: str, device: Any) -> str:
    if requested not in {"auto", "torch", "tilelang"}:
        raise ValueError(f"Unsupported kernel backend: {requested}")
    normalized = requested
    if normalized == "auto":
        device_type = getattr(device, "type", str(device).split(":", 1)[0])
        if device_type == "cuda" and tilelang_status().available:
            return "tilelang"
        return "torch"
    status = kernel_backend_status()[normalized]
    if not status.available:
        detail = f": {status.reason}" if status.reason else ""
        raise RuntimeError(f"Kernel backend is unavailable: {normalized}{detail}")
    return normalized
