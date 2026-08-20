from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from rwkv_publisher import conversion
from rwkv_publisher import source as source_module
from rwkv_publisher.assets import ASSET_SET, asset_root, verify_assets
from rwkv_publisher.profiles import (
    CheckpointProfile,
    FamilyProfile,
    ProfileRegistry,
    load_profiles,
)
from rwkv_publisher.runtime_export import (
    FLAT_RUNTIME_FILES,
    build_flat_runtime,
    transform_runtime_source,
)
from rwkv_publisher.source import resolve_source


def test_asset_lock_and_official_profiles_are_complete() -> None:
    lock = verify_assets()
    assert lock["asset_set"] == ASSET_SET
    assert all(
        item["version"] == ASSET_SET and item["origin"]
        for item in lock["files"].values()
    )
    profiles = load_profiles().checkpoints
    assert {
        profile.model_name: (profile.sha256, profile.size_bytes)
        for profile in profiles.values()
    } == {
        "RWKV7-1.5B-20260805": (
            "32ef7b5bf4dc8bde843cf26dfad809a1f527e2e76a9e790e7d406e71bcd785da",
            3_055_444_605,
        ),
        "RWKV7-0.1B-20241210": (
            "60c98129b9529963bff2c164b8ab4bd17c19332ae06dc2dcae32aa3a3739295a",
            382_195_690,
        ),
        "RWKV7-1.5B-20260710": (
            "737079d81865801fd85e5459488d89a36d5304a524e890244eb83d44f531c89c",
            3_055_444_605,
        ),
        "RWKV7-7.2B-20260710": (
            "1fe61e5c4b9037ffd4723a11c4de146d99c26bcd89e00a61afa67ef653d215e8",
            14_400_007_869,
        ),
    }


def test_local_source_resolution_rejects_known_name_with_wrong_hash(
    tmp_path: Path,
) -> None:
    profile = load_profiles().checkpoints["world-0.1b-20241210"]
    checkpoint = tmp_path / profile.filename
    checkpoint.write_bytes(b"wrong checkpoint")
    with pytest.raises(ValueError, match="known checkpoint filename"):
        resolve_source(checkpoint)


def test_unknown_local_source_omits_unverified_claims(tmp_path: Path) -> None:
    checkpoint = tmp_path / "custom-20260806.pth"
    checkpoint.write_bytes(b"checkpoint")
    result = resolve_source(checkpoint)
    assert result.kind == "local"
    assert result.profile is None
    assert result.reference is None
    assert result.sha256 == hashlib.sha256(b"checkpoint").hexdigest()


def test_unknown_source_reference_is_pinned_and_content_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "custom-20260806.pth"
    checkpoint.write_bytes(b"checkpoint")

    class Api:
        def model_info(self, repo_id: str):
            assert repo_id == "aabbdev/custom"
            return type("Info", (), {"sha": "b" * 40})()

    monkeypatch.setattr(source_module, "HfApi", Api)
    monkeypatch.setattr(
        source_module,
        "hf_hub_download",
        lambda **kwargs: str(checkpoint),
    )
    result = resolve_source(checkpoint, source_ref="aabbdev/custom/custom-20260806.pth")
    assert result.reference == "aabbdev/custom/custom-20260806.pth"
    assert result.revision == "b" * 40


def test_registered_hub_source_uses_locked_revision_and_offline_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blob = tmp_path / "blob"
    blob.write_bytes(b"registered")
    checkpoint = tmp_path / "model-20260806.pth"
    checkpoint.symlink_to(blob)
    digest = hashlib.sha256(blob.read_bytes()).hexdigest()
    profile = CheckpointProfile(
        id="test",
        family="test-family",
        filename=checkpoint.name,
        hub_repo="BlinkDL/test",
        hub_revision="a" * 40,
        sha256=digest,
        size_bytes=blob.stat().st_size,
        parameter_label="0.1",
        release_date="20260806",
        context_length=4096,
        license="apache-2.0",
    )
    registry = ProfileRegistry(
        families={
            "test-family": FamilyProfile(
                id="test-family", languages=(), datasets=(), chat_template="rwkv-world"
            )
        },
        checkpoints={"test": profile},
    )
    calls = []

    def download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(checkpoint)

    monkeypatch.setattr(source_module, "hf_hub_download", download)
    result = resolve_source("BlinkDL/test", registry=registry, offline=True)
    assert result.profile == profile
    assert result.local_path == blob.resolve()
    assert calls == [
        {
            "repo_id": "BlinkDL/test",
            "filename": checkpoint.name,
            "revision": "a" * 40,
            "local_files_only": True,
        }
    ]


def test_sha_matched_renamed_checkpoint_uses_canonical_identity(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "renamed.pth"
    checkpoint.write_bytes(b"official bytes")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    profile = CheckpointProfile(
        id="test",
        family="test-family",
        filename="canonical-20260806.pth",
        hub_repo="BlinkDL/test",
        hub_revision="c" * 40,
        sha256=digest,
        size_bytes=checkpoint.stat().st_size,
        parameter_label="0.1",
        release_date="20260806",
        context_length=4096,
        license="apache-2.0",
    )
    registry = ProfileRegistry(
        families={
            "test-family": FamilyProfile(
                id="test-family", languages=(), datasets=(), chat_template="rwkv-world"
            )
        },
        checkpoints={"test": profile},
    )
    result = resolve_source(checkpoint, registry=registry)
    assert result.local_path == checkpoint
    assert result.filename == profile.filename
    assert result.reference == profile.source_reference


def test_dtype_preservation_and_explicit_cast_contract() -> None:
    assert conversion._target_dtype(("bfloat16",), "preserve") == (
        "bfloat16",
        False,
    )
    assert conversion._target_dtype(("float32",), "float16") == ("float16", True)
    with pytest.raises(ValueError, match="mixed"):
        conversion._target_dtype(("bfloat16", "float32"), "preserve")


def test_flat_runtime_export_has_no_legacy_packages() -> None:
    runtime = asset_root() / "runtime" / ASSET_SET
    exported = build_flat_runtime(runtime)
    assert set(exported.files) == set(FLAT_RUNTIME_FILES)
    assert exported.provenance["format_version"] == 4
    for filename, source in exported.files.items():
        compile(source, filename, "exec")
        assert "from rwkv7_pytorch" not in source
        assert "from .kernel_tilelang" not in source


def test_runtime_transform_changes_only_allowlisted_import_span() -> None:
    source = (
        "from __future__ import annotations\n\n"
        "from .kernel_tilelang_state import cuda_arch_key\n\n"
        "MARKER = 'body bytes stay exact'\n"
    )
    transformed = transform_runtime_source(source, "sample.py")
    assert "from .kernel import state as _kernel_state" in transformed
    assert "cuda_arch_key = _kernel_state.cuda_arch_key" in transformed
    assert transformed.endswith("\n\nMARKER = 'body bytes stay exact'\n")
