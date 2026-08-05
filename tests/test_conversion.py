from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from transformers import AutoTokenizer, PreTrainedConfig, PreTrainedTokenizerFast

from rwkv_publisher import conversion
from rwkv_publisher.cli import main


def _vocab(path: Path) -> Path:
    rows = [f"{byte + 1} {bytes([byte])!r} 1" for byte in range(256)]
    rows.extend(["257 b'ab' 2", "258 b'abc' 3"])
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _tiny_native(torch: Any) -> dict[str, Any]:
    config = {
        "vocab_size": 259,
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "num_heads": 1,
        "head_dim": 64,
        "intermediate_size": 128,
        "decay_low_rank_dim": 8,
        "a_low_rank_dim": 8,
        "v_low_rank_dim": 4,
        "gate_low_rank_dim": 16,
    }
    return {
        key.removeprefix("rwkv7."): torch.zeros(shape, dtype=torch.float32)
        for key, shape in conversion._expected_shapes(config).items()
    }


def test_conversion_helpers_reject_unknown_keys_and_bad_sizes() -> None:
    assert conversion._size_bytes("1GB") == 1_000_000_000
    assert conversion._size_bytes("1GiB") == 1 << 30
    assert conversion._native_key("blocks.0.att.r_k") == "rwkv7.blocks.0.att.r_k"
    with pytest.raises(KeyError, match="unrecognized"):
        conversion._native_key("unexpected.weight")
    with pytest.raises(ValueError, match="invalid shard size"):
        conversion._size_bytes("zero")


def test_conversion_rejects_non_floating_model_tensors() -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="non-floating"):
        conversion._source_float_dtypes({"weight": torch.zeros(2, dtype=torch.int64)})


def test_convert_checkpoint_writes_native_sharded_model(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    checkpoint = tmp_path / "model.pth"
    torch.save({"state_dict": _tiny_native(torch)}, checkpoint)
    output = tmp_path / "hf"

    result = conversion.convert_checkpoint(
        checkpoint,
        output,
        vocab_file=_vocab(tmp_path / "rwkv_vocab_v20230424.txt"),
        dtype="bfloat16",
        max_shard_size="40KB",
    )

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    index = json.loads(
        (output / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    assert config["architectures"] == ["Rwkv7ForCausalLM"]
    assert config["model_type"] == "rwkv7"
    assert config["hidden_size"] == 64
    assert config["num_heads"] == 1
    assert config["head_dim"] == 64
    assert config["dtype"] == "bfloat16"
    assert set(index["weight_map"]) == {
        conversion._native_key(key) for key in _tiny_native(torch)
    }
    assert result.tensor_count == len(index["weight_map"])
    assert result.source_float_dtypes == ("float32",)
    assert result.target_float_dtype == "bfloat16"
    assert result.explicit_cast is True
    assert len(result.weight_files) > 1
    assert (output / "tokenizer.json").is_file()
    assert not (output / "rwkv_vocab_v20230424.txt").exists()
    assert not (output / "vocab.json").exists()
    tokenizer = AutoTokenizer.from_pretrained(
        output, config=PreTrainedConfig(), local_files_only=True
    )
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    assert tokenizer.is_fast
    assert tokenizer.encode("abc", add_special_tokens=False) == [258]


def test_shared_storage_sharding_honors_logical_limit(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    storage = torch.arange(3072, dtype=torch.float32)
    state = {
        "first": storage[:1024],
        "second": storage[1024:2048],
        "third": storage[2048:],
    }

    files, logical_size = conversion._save_sharded_safetensors(
        state, tmp_path, "6KB", "float32"
    )

    index = json.loads(
        (tmp_path / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    assert len(files) == 3
    assert logical_size == 3072 * 4
    assert len(set(index["weight_map"].values())) == 3


def test_inferred_config_rejects_missing_and_wrong_shape_tensors() -> None:
    torch = pytest.importorskip("torch")
    converted = conversion._convert_native(_tiny_native(torch))
    missing = dict(converted)
    missing.pop("rwkv7.blocks.0.att.output.weight")
    with pytest.raises(ValueError, match="tensor keys mismatch"):
        conversion._infer_config(missing, "float32")

    wrong = dict(converted)
    wrong["rwkv7.blocks.0.ffn.value.weight"] = torch.zeros(63, 128)
    with pytest.raises(ValueError, match="tensor shapes mismatch"):
        conversion._infer_config(wrong, "float32")


def test_convert_refuses_existing_output_and_symlinked_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    link = tmp_path / "link.pth"
    link.symlink_to(checkpoint)
    vocab = _vocab(tmp_path / "rwkv_vocab_v20230424.txt")
    with pytest.raises(FileNotFoundError, match="checkpoint does not exist"):
        conversion.convert_checkpoint(link, tmp_path / "out", vocab_file=vocab)
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        conversion.convert_checkpoint(checkpoint, output, vocab_file=vocab)


def test_build_cli_dispatches_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, Any] = {}

    def fake_build(source: str, **kwargs: Any) -> dict[str, Any]:
        captured.update(source=source, **kwargs)
        return {"dry_run": True, "source": source, "output": str(kwargs["output"])}

    monkeypatch.setattr("rwkv_publisher.cli.build_release", fake_build)
    main(
        [
            "build",
            str(tmp_path / "model.pth"),
            "--output",
            str(tmp_path / "dist"),
            "--dtype",
            "float16",
            "--max-shard-size",
            "1GB",
            "--config",
            str(tmp_path / "metadata.toml"),
            "--profile",
            "world-v2.8",
            "--no-input",
            "--json",
        ]
    )

    assert captured["dtype"] == "float16"
    assert captured["max_shard_size"] == "1GB"
    assert captured["metadata_config"] == tmp_path / "metadata.toml"
    assert captured["metadata_profile"] == "world-v2.8"
    assert captured["interactive"] is False
    assert json.loads(capsys.readouterr().out)["output"].endswith("/dist")
