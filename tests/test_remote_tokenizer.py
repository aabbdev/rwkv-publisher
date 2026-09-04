from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import AutoTokenizer

from rwkv_publisher.assets import asset_path
from rwkv_publisher.encoding import (
    build_fast_tokenizer,
    encode_reference,
    read_rwkv_vocab,
)
from rwkv_publisher.remote_code import TOKENIZER_AUTO_MAP, build_model_code


def _tiny_vocab(path: Path) -> Path:
    rows = [f"{byte + 1} {bytes([byte])!r} 1" for byte in range(256)]
    rows.extend(["257 b'ab' 2", "258 b'abc' 3", "259 b'  ' 2"])
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _root(tmp_path: Path, vocab_file: Path, vocab_size: int) -> Path:
    build_fast_tokenizer(vocab_file, vocab_size).save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenization_rwkv7.py").write_text(
        build_model_code()["tokenization_rwkv7.py"], encoding="utf-8"
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "auto_map": TOKENIZER_AUTO_MAP,
                "backend": "rwkv_world_trie",
                "eos_token": "<|endoftext|>",
                "pad_token": "<|endoftext|>",
                "padding_side": "left",
                "tokenizer_class": "Rwkv7Tokenizer",
                "unk_token": "<|endoftext|>",
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def _load(root: Path) -> Any:
    return AutoTokenizer.from_pretrained(
        root, local_files_only=True, trust_remote_code=True
    )


def test_auto_tokenizer_loads_remote_trie_and_hf_batch(tmp_path: Path) -> None:
    tokenizer = _load(_root(tmp_path, _tiny_vocab(tmp_path / "vocab.txt"), 262))
    assert tokenizer.__class__.__name__ == "Rwkv7Tokenizer"
    assert tokenizer.__class__.__module__.startswith("transformers_modules.")
    assert not tokenizer.is_fast
    assert tokenizer.encode("abcab a", add_special_tokens=False) == [258, 257, 33, 98]
    batch = tokenizer(
        ["abc", "a a"], padding=True, return_tensors="pt", add_special_tokens=False
    )
    assert batch["input_ids"].dtype == torch.int64
    assert batch["input_ids"].tolist() == [[0, 0, 258], [98, 33, 98]]
    assert batch["attention_mask"].tolist() == [[0, 0, 1], [1, 1, 1]]


def test_remote_trie_matches_rwkv_oracle_on_official_long_inputs(
    tmp_path: Path,
) -> None:
    vocab_file = asset_path("vocab/rwkv_vocab_v20230424.txt")
    tokens = read_rwkv_vocab(vocab_file)
    tokenizer = _load(_root(tmp_path, vocab_file, 65_536))
    random_source = random.Random(7)
    alphabet = " abcXYZ0123\n\t.,—é東🙂"
    samples = [
        "hello  world",
        "l’amour…\n東京🙂",
        "abc123" * 2_000,
        " x" * 8_192,
        *(
            "".join(random_source.choice(alphabet) for _ in range(128))
            for _ in range(50)
        ),
    ]
    for text in samples:
        expected = encode_reference(text, tokens)
        assert tokenizer.encode(text, add_special_tokens=False) == expected
        assert tokenizer.decode(expected, skip_special_tokens=False) == text


def test_remote_trie_encodes_every_official_token_to_its_id(tmp_path: Path) -> None:
    vocab_file = asset_path("vocab/rwkv_vocab_v20230424.txt")
    tokens = read_rwkv_vocab(vocab_file)
    tokenizer = _load(_root(tmp_path, vocab_file, 65_536))
    for token_id, token in tokens.items():
        assert tokenizer._encode_bytes(token) == [token_id]


def test_remote_tokenizer_rejects_bos_and_tampered_semantics(tmp_path: Path) -> None:
    root = _root(tmp_path, _tiny_vocab(tmp_path / "vocab.txt"), 262)
    with pytest.raises(ValueError, match="no BOS"):
        AutoTokenizer.from_pretrained(
            root,
            local_files_only=True,
            trust_remote_code=True,
            add_bos_token=True,
        )
    value = json.loads((root / "tokenizer.json").read_text(encoding="utf-8"))
    value["pre_tokenizer"]["use_regex"] = True
    (root / "tokenizer.json").write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="ByteLevel"):
        AutoTokenizer.from_pretrained(
            root, local_files_only=True, trust_remote_code=True
        )


def test_locked_remote_tokenizer_has_no_eval_or_external_vocab_dependency() -> None:
    source = build_model_code()["tokenization_rwkv7.py"]
    assert "eval(" not in source
    assert "rwkv_vocab" not in source
    assert '"tokenizer.json"' in source
