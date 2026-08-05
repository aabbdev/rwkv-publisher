from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest
from transformers import AutoTokenizer, PreTrainedConfig

from rwkv_publisher import build as build_module
from rwkv_publisher import conversion
from rwkv_publisher import manifest as manifest_module
from rwkv_publisher.assets import sha256_file
from rwkv_publisher.build import build_release
from rwkv_publisher.manifest import validate_release
from rwkv_publisher.profiles import load_profiles
from rwkv_publisher.source import ResolvedSource


def _vocab(path: Path) -> Path:
    rows = [f"{byte + 1} {bytes([byte])!r} 1" for byte in range(256)]
    rows.extend(["257 b'ab' 2", "258 b'abc' 3"])
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _checkpoint(path: Path) -> Path:
    torch = pytest.importorskip("torch")
    config = {
        "vocab_size": 259,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_heads": 1,
        "head_dim": 64,
        "intermediate_size": 128,
        "decay_low_rank_dim": 8,
        "a_low_rank_dim": 8,
        "v_low_rank_dim": 4,
        "gate_low_rank_dim": 16,
    }
    native = {
        key.removeprefix("rwkv7."): torch.zeros(shape, dtype=torch.float32)
        for key, shape in conversion._expected_shapes(config).items()
        if key
        not in {
            "rwkv7.blocks.0.att.v0",
            "rwkv7.blocks.0.att.v1",
            "rwkv7.blocks.0.att.v2",
        }
    }
    torch.save(native, path)
    return path


@pytest.fixture
def built_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    vocab = _vocab(tmp_path / "vocab.txt")
    monkeypatch.setattr(conversion, "asset_path", lambda _: vocab)
    locked_asset_path = manifest_module.asset_path
    monkeypatch.setattr(
        manifest_module,
        "asset_path",
        lambda relative: (
            vocab
            if relative == "vocab/rwkv_vocab_v20230424.txt"
            else locked_asset_path(relative)
        ),
    )
    monkeypatch.setattr(build_module, "_parameter_label", lambda _: "0.1")
    checkpoint = _checkpoint(tmp_path / "rwkv-world-20241210.pth")
    result = build_release(checkpoint, output=tmp_path / "dist", max_shard_size="1GB")
    return Path(result["release"])


def test_build_produces_native_flat_valid_release(built_release: Path) -> None:
    manifest = validate_release(built_release)
    assert manifest["schema_version"] == 3
    assert manifest["identity"]["parameter_label"] == "0.1"
    assert manifest["conversion"]["source_float_dtypes"] == ["float32"]
    assert manifest["conversion"]["target_float_dtypes"] == ["float32"]
    assert manifest["conversion"]["explicit_cast"] is False
    assert manifest["conversion"]["synthesized_tensors"] == [
        "rwkv7.blocks.0.att.v0",
        "rwkv7.blocks.0.att.v1",
        "rwkv7.blocks.0.att.v2",
    ]
    assert (built_release / "inference/kernel.py").is_file()
    assert not (built_release / "inference/decode").exists()
    assert not (built_release / "inference/rwkv7_pytorch").exists()
    assert not list(built_release.glob("*.py"))
    for path in (built_release / "inference").glob("*.py"):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    tokenizer = AutoTokenizer.from_pretrained(
        built_release, config=PreTrainedConfig(), local_files_only=True
    )
    assert tokenizer.encode("abc", add_special_tokens=False) == [258]
    assert (built_release.stat().st_mode & 0o222) == 0
    assert (built_release / "README.md").stat().st_mode & 0o222 == 0


def test_flat_runtime_imports_with_legacy_packages_blocked(
    built_release: Path,
) -> None:
    script = """
import importlib.abc
import sys

class BlockLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "rwkv7_pytorch" or fullname.startswith("rwkv7_pytorch."):
            raise ImportError(f"legacy package import attempted: {fullname}")
        if fullname == "decode" or fullname.startswith("decode."):
            raise ImportError(f"legacy package import attempted: {fullname}")
        return None

sys.meta_path.insert(0, BlockLegacy())
from inference.runtime import RWKV7Config
assert RWKV7Config.model_type == "rwkv7"
print(sys.modules["inference.runtime"].__file__)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=built_release,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert str(built_release / "inference/runtime.py") in result.stdout


def test_compact_runtime_direct_cli_avoids_package_collisions(
    built_release: Path,
) -> None:
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    collision = built_release.parent.parent / "collision"
    collision.mkdir()
    collision.joinpath("inference.py").write_text(
        "raise RuntimeError('unrelated inference package imported')\n",
        encoding="utf-8",
    )
    environment["PYTHONPATH"] = str(collision)
    direct = subprocess.run(
        [sys.executable, "inference/generate.py", "--help"],
        cwd=built_release,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert direct.returncode == 0, direct.stderr
    assert "--backend" in direct.stdout


def test_build_output_is_deterministic(
    built_release: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(build_module, "_parameter_label", lambda _: "0.1")
    source = built_release.parent.parent / "rwkv-world-20241210.pth"
    second = Path(
        build_release(source, output=built_release.parent.parent / "second")["release"]
    )
    first_files = {
        path.relative_to(built_release).as_posix(): path.read_bytes()
        for path in built_release.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second).as_posix(): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
    assert second_files == first_files


def test_build_is_atomic_and_refuses_overwrite(
    built_release: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(build_module, "_parameter_label", lambda _: "0.1")
    source = built_release.parent.parent / "rwkv-world-20241210.pth"
    with pytest.raises(FileExistsError, match="overwrite"):
        build_release(source, output=built_release.parent)
    assert not list(built_release.parent.glob(".rwkv-publisher-*"))


def test_build_rejects_source_mutation_during_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vocab = _vocab(tmp_path / "vocab.txt")
    monkeypatch.setattr(conversion, "asset_path", lambda _: vocab)
    monkeypatch.setattr(build_module, "_parameter_label", lambda _: "0.1")
    checkpoint = _checkpoint(tmp_path / "custom-20260806.pth")
    real_convert = build_module.convert_into

    def mutating_convert(*args, **kwargs):
        result = real_convert(*args, **kwargs)
        checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
        return result

    monkeypatch.setattr(build_module, "convert_into", mutating_convert)
    with pytest.raises(RuntimeError, match="source checkpoint changed"):
        build_release(checkpoint, output=tmp_path / "dist")
    assert not list((tmp_path / "dist").iterdir())


def test_manifest_detects_tampering(built_release: Path) -> None:
    (built_release / "README.md").chmod(0o644)
    (built_release / "README.md").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="(size|SHA-256) mismatch"):
        validate_release(built_release)


def _accept_tampered_hash(root: Path, relative: str) -> dict:
    manifest_path = root / "release-manifest.json"
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = root / relative
    manifest["files"][relative]["size_bytes"] = path.stat().st_size
    manifest["files"][relative]["sha256"] = sha256_file(path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def test_manifest_reconstructs_locked_runtime_independently(
    built_release: Path,
) -> None:
    relative = "inference/runtime.py"
    path = built_release / relative
    path.chmod(0o644)
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    manifest = _accept_tampered_hash(built_release, relative)
    manifest["runtime"]["files"]["runtime.py"]["sha256"] = sha256_file(path)
    (built_release / "release-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="locked runtime"):
        validate_release(built_release)


def test_manifest_reconstructs_locked_tokenizer_independently(
    built_release: Path,
) -> None:
    relative = "tokenizer.json"
    path = built_release / relative
    path.chmod(0o644)
    tokenizer = json.loads(path.read_text(encoding="utf-8"))
    vocab = tokenizer["model"]["vocab"]
    first, second = [key for key, value in vocab.items() if value not in {0}][:2]
    vocab[first], vocab[second] = vocab[second], vocab[first]
    path.write_text(json.dumps(tokenizer), encoding="utf-8")
    _accept_tampered_hash(built_release, relative)
    with pytest.raises(ValueError, match="locked vocabulary"):
        validate_release(built_release)


def test_manifest_reconstructs_model_card_and_license(built_release: Path) -> None:
    card = built_release / "README.md"
    card.chmod(0o644)
    card.write_text(card.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    _accept_tampered_hash(built_release, "README.md")
    with pytest.raises(ValueError, match="model card does not match"):
        validate_release(built_release)


def test_known_profile_dry_run_does_not_create_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = load_profiles().checkpoints["world-0.1b-20241210"]
    source = tmp_path / profile.filename
    source.write_bytes(b"not read during dry run")
    monkeypatch.setattr(
        build_module,
        "resolve_source",
        lambda *args, **kwargs: ResolvedSource(
            kind="hub",
            local_path=source,
            filename=profile.filename,
            sha256=profile.sha256,
            size_bytes=profile.size_bytes,
            reference=f"{profile.hub_repo}/{profile.filename}",
            revision=profile.hub_revision,
            profile=profile,
        ),
    )
    monkeypatch.setattr(
        build_module,
        "inspect_checkpoint",
        lambda *args, **kwargs: conversion.ConversionResult(
            config={},
            source_float_dtypes=("bfloat16",),
            target_float_dtype="bfloat16",
            explicit_cast=False,
            source_parameter_count=191_084_544,
            serialized_parameter_count=191_084_544,
            tensor_count=402,
            synthesized_tensors=(),
            weight_bytes=382_169_088,
            weight_files=(),
        ),
    )
    result = build_release(
        "BlinkDL/rwkv-7-world/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth",
        output=tmp_path / "dist",
        offline=True,
        dry_run=True,
    )
    assert result["dry_run"] is True
    assert result["model_name"] == "RWKV7-0.1B-20241210"
    assert not (tmp_path / "dist").exists()


def test_unknown_dry_run_inspects_weights_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(build_module, "_parameter_label", lambda _: "0.1")
    checkpoint = _checkpoint(tmp_path / "custom-20260806.pth")
    result = build_release(checkpoint, output=tmp_path / "dist", dry_run=True)
    assert result["model_name"] == "RWKV7-0.1B-20260806"
    assert result["dtype"] == "float32"
    assert result["tensor_count"] > 0
    assert not (tmp_path / "dist").exists()

    corrupt = tmp_path / "corrupt-20260806.pth"
    corrupt.write_bytes(b"not a checkpoint")
    with pytest.raises((EOFError, RuntimeError, ValueError, pickle.UnpicklingError)):
        build_release(corrupt, output=tmp_path / "unused", dry_run=True)


def test_release_manifest_is_destination_neutral(built_release: Path) -> None:
    manifest = json.loads(
        (built_release / "release-manifest.json").read_text(encoding="utf-8")
    )
    assert "repo_id" not in json.dumps(manifest)
