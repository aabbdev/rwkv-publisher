from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from rwkv_publisher import metadata as metadata_module
from rwkv_publisher.metadata import load_metadata_config, resolve_release_metadata
from rwkv_publisher.profiles import (
    CheckpointProfile,
    FamilyProfile,
    ProfileRegistry,
)
from rwkv_publisher.source import ResolvedSource


class TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def _registry() -> ProfileRegistry:
    return ProfileRegistry(
        families={
            "alpha": FamilyProfile(
                id="alpha",
                languages=("en", "fr"),
                datasets=("owner/alpha",),
                chat_template="rwkv-world",
                context_length=4096,
                license="apache-2.0",
            ),
            "beta": FamilyProfile(
                id="beta",
                languages=("zh",),
                datasets=("owner/beta",),
                chat_template="rwkv-world",
                context_length=8192,
                license="mit",
            ),
        },
        checkpoints={},
    )


def _source(tmp_path: Path, profile: CheckpointProfile | None = None) -> ResolvedSource:
    path = tmp_path / "model-20260806.pth"
    path.write_bytes(b"checkpoint")
    return ResolvedSource(
        kind="local",
        local_path=path,
        filename=profile.filename if profile else path.name,
        sha256="a" * 64,
        size_bytes=path.stat().st_size,
        reference=profile.source_reference if profile else None,
        revision=profile.hub_revision if profile else None,
        profile=profile,
    )


def test_toml_profile_and_custom_metadata_are_minimal(tmp_path: Path) -> None:
    config = tmp_path / "metadata.toml"
    config.write_text(
        'schema = 1\nprofile = "alpha"\n\n[metadata]\n'
        'languages = ["de"]\ncontext_length = 2048\nlicense = "mit"\n',
        encoding="utf-8",
    )
    parsed = load_metadata_config(config)
    assert parsed.profile == "alpha"
    result = resolve_release_metadata(
        _source(tmp_path), config_path=config, registry=_registry()
    )
    assert result.profile == "alpha"
    assert result.languages == ("de",)
    assert result.datasets == ("owner/alpha",)
    assert result.context_length == 2048
    assert result.license == "mit"
    assert result.provenance == "config"


def test_profile_only_toml_remains_lock_derived(tmp_path: Path) -> None:
    config = tmp_path / "metadata.toml"
    config.write_text('schema = 1\nprofile = "alpha"\n', encoding="utf-8")
    result = resolve_release_metadata(
        _source(tmp_path), config_path=config, registry=_registry()
    )
    assert result.languages == ("en", "fr")
    assert result.datasets == ("owner/alpha",)
    assert result.context_length == 4096
    assert result.license == "apache-2.0"
    assert result.provenance == "config-profile"


def test_toml_rejects_technical_build_fields(tmp_path: Path) -> None:
    config = tmp_path / "metadata.toml"
    config.write_text('schema = 1\ndtype = "bfloat16"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported top-level"):
        load_metadata_config(config)


@pytest.mark.parametrize("schema", ["true", "1.0", '"1"'])
def test_toml_requires_integer_schema(tmp_path: Path, schema: str) -> None:
    config = tmp_path / "metadata.toml"
    config.write_text(f"schema = {schema}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema must be 1"):
        load_metadata_config(config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("context_length", "true", "positive integer"),
        ("context_length", "0", "positive integer"),
        ("license", '"not a license"', "SPDX-style"),
    ],
)
def test_toml_validates_context_and_license(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    config = tmp_path / "metadata.toml"
    config.write_text(f"schema = 1\n[metadata]\n{field} = {value}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_metadata_config(config)


def test_recognized_checkpoint_rejects_incompatible_profile(tmp_path: Path) -> None:
    profile = CheckpointProfile(
        id="checkpoint",
        family="alpha",
        filename="model-20260806.pth",
        hub_repo="owner/repo",
        hub_revision="b" * 40,
        sha256="a" * 64,
        size_bytes=10,
        parameter_label="0.1",
        release_date="20260806",
        context_length=4096,
        license="apache-2.0",
    )
    with pytest.raises(ValueError, match="requires metadata profile"):
        resolve_release_metadata(
            _source(tmp_path, profile),
            profile_override="beta",
            registry=_registry(),
        )


def test_recognized_checkpoint_rejects_context_or_license_override(
    tmp_path: Path,
) -> None:
    profile = CheckpointProfile(
        id="checkpoint",
        family="alpha",
        filename="model-20260806.pth",
        hub_repo="owner/repo",
        hub_revision="b" * 40,
        sha256="a" * 64,
        size_bytes=10,
        parameter_label="0.1",
        release_date="20260806",
        context_length=4096,
        license="apache-2.0",
    )
    config = tmp_path / "metadata.toml"
    config.write_text(
        'schema = 1\n[metadata]\ncontext_length = 8192\nlicense = "mit"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must match its locked profile"):
        resolve_release_metadata(
            _source(tmp_path, profile), config_path=config, registry=_registry()
        )


def test_unknown_checkpoint_prompts_only_on_a_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    terminal = SimpleNamespace(isatty=lambda: True)
    stdout = TTYBuffer()
    stderr = TTYBuffer()
    monkeypatch.setattr(
        metadata_module,
        "sys",
        SimpleNamespace(stdin=terminal, stdout=stdout, stderr=stderr),
    )
    monkeypatch.setattr("builtins.input", lambda: "1")
    result = resolve_release_metadata(
        _source(tmp_path), interactive=True, registry=_registry()
    )
    assert result.profile == "alpha"
    assert result.languages == ("en", "fr")
    assert result.context_length == 4096
    assert result.license == "apache-2.0"
    assert result.provenance == "interactive"
    assert "No metadata profile matched" in stderr.getvalue()
    assert "Select metadata profile" in stderr.getvalue()
    assert stdout.getvalue() == ""


def test_explicit_empty_config_suppresses_terminal_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "metadata.toml"
    config.write_text("schema = 1\n", encoding="utf-8")
    terminal = SimpleNamespace(isatty=lambda: True)
    monkeypatch.setattr(
        metadata_module,
        "sys",
        SimpleNamespace(stdin=terminal, stdout=terminal, stderr=TTYBuffer()),
    )
    monkeypatch.setattr(
        "builtins.input",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected prompt")),
    )
    result = resolve_release_metadata(
        _source(tmp_path),
        config_path=config,
        interactive=True,
        registry=_registry(),
    )
    assert result.provenance == "none"
    assert result.profile is None


def test_unknown_noninteractive_checkpoint_has_no_claims(tmp_path: Path) -> None:
    result = resolve_release_metadata(
        _source(tmp_path), interactive=False, registry=_registry()
    )
    assert result.profile is None
    assert result.languages == ()
    assert result.datasets == ()
    assert result.context_length is None
    assert result.license is None
    assert result.provenance == "none"
