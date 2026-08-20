from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "src/rwkv_publisher/assets"
LOCK = ASSETS / "assets.lock.json"
ASSET_SET = "2026.08.06-r1"
MODEL_CODE_REPOSITORY = "https://github.com/huggingface/transformers.git"
MODEL_CODE_REVISION = "4ad9ed0747ed6ba75c787e8f9040dcd64b166ee2"
MODEL_CODE_SOURCE_DIRECTORY = "src/transformers/models/rwkv7"
MODEL_CODE_FILENAMES = ("configuration_rwkv7.py", "modeling_rwkv7.py")


def origin(relative: str) -> str:
    if relative == "profiles.json":
        return "verified BlinkDL Hub checkpoint metadata"
    if relative == "vocab/rwkv_vocab_v20230424.txt":
        return "RWKV7_Pytorch/rwkv_vocab_v20230424.txt"
    if relative == "templates/chat_template.jinja":
        return "RWKV7_Pytorch/chat_template.j2"
    if relative.startswith("model_code/"):
        if relative not in {
            f"model_code/{filename}" for filename in MODEL_CODE_FILENAMES
        }:
            raise ValueError(f"asset has no declared origin: {relative}")
        filename = Path(relative).name
        return (
            f"{MODEL_CODE_REPOSITORY}@{MODEL_CODE_REVISION}/"
            f"{MODEL_CODE_SOURCE_DIRECTORY}/{filename}"
        )
    if relative.startswith(f"runtime/{ASSET_SET}/"):
        return f"RWKV7_Pytorch/rwkv7_pytorch/{Path(relative).name}"
    raise ValueError(f"asset has no declared origin: {relative}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    files = {}
    for path in sorted(ASSETS.rglob("*")):
        if path.is_file() and path != LOCK:
            relative = path.relative_to(ASSETS).as_posix()
            files[relative] = {
                "origin": origin(relative),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
                "version": ASSET_SET,
            }
    combined = hashlib.sha256()
    for name, item in files.items():
        combined.update(
            f"{name}\0{json.dumps(item, sort_keys=True, separators=(',', ':'))}\n".encode()
        )
    document = {
        "schema_version": 1,
        "asset_set": ASSET_SET,
        "combined_sha256": combined.hexdigest(),
        "compatibility": {
            "cuda": "12.8",
            "python": "3.12",
            "tilelang": "0.1.12",
            "torch": "2.8.x",
            "transformers": ">=5.15",
        },
        "files": files,
    }
    LOCK.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
