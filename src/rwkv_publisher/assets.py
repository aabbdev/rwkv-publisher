from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
from pathlib import Path, PurePosixPath
from typing import Any

ASSET_SET = "2026.08.06-r1"


def asset_root() -> Path:
    return Path(__file__).resolve().parent / "assets"


def asset_path(relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
        raise ValueError(f"unsafe asset path: {relative!r}")
    path = asset_root().joinpath(*pure.parts)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"missing publisher asset: {relative}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_license_path() -> Path:
    local = Path(__file__).resolve().parents[2] / "LICENSE"
    if local.is_file():
        return local
    for item in importlib_metadata.files("rwkv-publisher") or ():
        if item.parts[-2:] == ("licenses", "LICENSE"):
            path = Path(str(item.locate()))
            if path.is_file():
                return path
    raise FileNotFoundError("rwkv-publisher canonical LICENSE is unavailable")


def verify_assets() -> dict[str, Any]:
    lock_path = asset_path("assets.lock.json")
    document = json.loads(lock_path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or document.get("asset_set") != ASSET_SET:
        raise ValueError("unsupported publisher asset lock")
    files = document.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("publisher asset lock has no files")
    actual = {
        path.relative_to(asset_root()).as_posix()
        for path in asset_root().rglob("*")
        if path.is_file() and path.name != "assets.lock.json"
    }
    if actual != set(files):
        raise ValueError("publisher asset inventory does not match lock")
    combined = hashlib.sha256()
    for relative, item in sorted(files.items()):
        path = asset_path(relative)
        if not isinstance(item, dict):
            raise TypeError(f"invalid asset lock entry: {relative}")
        if not isinstance(item.get("origin"), str) or item.get("version") != ASSET_SET:
            raise ValueError(f"publisher asset provenance is invalid: {relative}")
        size = item.get("size_bytes")
        digest = item.get("sha256")
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise ValueError(f"publisher asset hash mismatch: {relative}")
        combined.update(
            f"{relative}\0{json.dumps(item, sort_keys=True, separators=(',', ':'))}\n".encode()
        )
    if combined.hexdigest() != document.get("combined_sha256"):
        raise ValueError("publisher asset combined hash mismatch")
    return document


__all__ = [
    "ASSET_SET",
    "asset_path",
    "asset_root",
    "canonical_license_path",
    "sha256_file",
    "verify_assets",
]
