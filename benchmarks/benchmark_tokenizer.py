from __future__ import annotations

import argparse
import json
import runpy
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

from rwkv_publisher.assets import asset_path
from rwkv_publisher.encoding import (
    build_fast_tokenizer,
    encode_reference,
    read_rwkv_vocab,
)


def _duration(callable_: Callable[[], list[int]]) -> tuple[float, list[int]]:
    start = time.perf_counter()
    result = callable_()
    return time.perf_counter() - start, result


def _remote_module() -> ModuleType:
    path = asset_path("model_code/tokenization_rwkv7.py")
    module = ModuleType("rwkv7_benchmark_tokenizer")
    module.__dict__.update(runpy.run_path(str(path)))
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, action="append")
    parser.add_argument("--include-wordpiece", action="store_true")
    args = parser.parse_args()
    sizes = args.size or [1_024, 4_096, 16_384, 65_536]
    vocab_file = asset_path("vocab/rwkv_vocab_v20230424.txt")
    tokens = read_rwkv_vocab(vocab_file)
    wordpiece = build_fast_tokenizer(vocab_file, 65_536)
    module = _remote_module()
    tokenizer_type: Any = module.Rwkv7Tokenizer
    rows = []
    with tempfile.TemporaryDirectory(prefix="rwkv-tokenizer-benchmark-") as temporary:
        tokenizer_file = Path(temporary) / "tokenizer.json"
        wordpiece.save(str(tokenizer_file))
        tokenizer = tokenizer_type(str(tokenizer_file))
        for pattern in ("a", "abc123", " x", "\n"):
            for size in sizes:
                text = (pattern * ((size + len(pattern) - 1) // len(pattern)))[:size]
                byte_size = len(text.encode())
                elapsed, token_ids = _duration(
                    lambda text=text: tokenizer.encode(text, add_special_tokens=False)
                )
                if token_ids != encode_reference(text, tokens):
                    raise RuntimeError("remote trie differs from the RWKV oracle")
                row = {
                    "implementation": "rwkv_remote_trie",
                    "pattern": repr(pattern),
                    "bytes": byte_size,
                    "tokens": len(token_ids),
                    "seconds": elapsed,
                    "mib_per_second": byte_size / elapsed / (1024 * 1024),
                }
                rows.append(row)
                if args.include_wordpiece:
                    elapsed, rust_ids = _duration(
                        lambda text=text: wordpiece.encode(text).ids
                    )
                    if rust_ids != token_ids:
                        raise RuntimeError("WordPiece differs from the RWKV trie")
                    rows.append(
                        {
                            **row,
                            "implementation": "rust_wordpiece",
                            "seconds": elapsed,
                            "mib_per_second": byte_size / elapsed / (1024 * 1024),
                        }
                    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
