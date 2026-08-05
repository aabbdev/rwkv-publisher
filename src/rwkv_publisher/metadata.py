from __future__ import annotations

import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .profiles import ProfileRegistry, load_profiles
from .source import ResolvedSource


@dataclass(frozen=True)
class MetadataConfig:
    profile: str | None = None
    languages: tuple[str, ...] | None = None
    datasets: tuple[str, ...] | None = None
    context_length: int | None = None
    license: str | None = None


@dataclass(frozen=True)
class ReleaseMetadata:
    profile: str | None
    languages: tuple[str, ...]
    datasets: tuple[str, ...]
    context_length: int | None
    license: str | None
    provenance: str


def _optional_strings(value: Any, field: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise TypeError(f"metadata {field} must be a list of non-empty strings")
    normalized = tuple(item.strip() for item in value)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"metadata {field} contains duplicates")
    return normalized


def load_metadata_config(path: Path) -> MetadataConfig:
    path = path.expanduser()
    if path.is_symlink() or not path.is_file() or path.suffix.lower() != ".toml":
        raise FileNotFoundError(f"metadata config must be a regular .toml file: {path}")
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    schema = document.get("schema")
    if type(schema) is not int or schema != 1:
        raise ValueError("metadata config schema must be 1")
    if set(document) - {"schema", "profile", "metadata"}:
        raise ValueError("metadata config contains unsupported top-level fields")
    profile = document.get("profile")
    if profile is not None and (not isinstance(profile, str) or not profile.strip()):
        raise TypeError("metadata profile must be a non-empty string")
    metadata = document.get("metadata", {})
    if not isinstance(metadata, dict) or set(metadata) - {
        "languages",
        "datasets",
        "context_length",
        "license",
    }:
        raise ValueError("metadata config contains unsupported metadata fields")
    context_length = metadata.get("context_length")
    if context_length is not None and (
        type(context_length) is not int or context_length <= 0
    ):
        raise ValueError("metadata context_length must be a positive integer")
    license_id = metadata.get("license")
    if license_id is not None and (
        not isinstance(license_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+-]*", license_id.strip()) is None
    ):
        raise ValueError("metadata license must be a simple SPDX-style identifier")
    return MetadataConfig(
        profile=profile.strip() if isinstance(profile, str) else None,
        languages=_optional_strings(metadata.get("languages"), "languages"),
        datasets=_optional_strings(metadata.get("datasets"), "datasets"),
        context_length=context_length,
        license=license_id.strip() if isinstance(license_id, str) else None,
    )


def _choose_profile(registry: ProfileRegistry) -> str | None:
    profiles = sorted(registry.families.values(), key=lambda item: item.id)
    print("\nNo metadata profile matched this checkpoint.", file=sys.stderr)
    print("  0. Publish without metadata claims (recommended)", file=sys.stderr)
    for index, profile in enumerate(profiles, start=1):
        print(
            f"  {index}. {profile.id} "
            f"({len(profile.languages)} languages, {len(profile.datasets)} datasets, "
            f"context {profile.context_length or 'unknown'}, "
            f"license {profile.license or 'unknown'})",
            file=sys.stderr,
        )
    while True:
        try:
            print("Select metadata profile [0]: ", end="", file=sys.stderr, flush=True)
            answer = input().strip()
        except EOFError:
            return None
        if answer in {"", "0"}:
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(profiles):
            return profiles[int(answer) - 1].id
        print(f"Choose a number from 0 to {len(profiles)}.", file=sys.stderr)


def resolve_release_metadata(
    source: ResolvedSource,
    *,
    config_path: Path | None = None,
    profile_override: str | None = None,
    interactive: bool = False,
    registry: ProfileRegistry | None = None,
) -> ReleaseMetadata:
    registry = registry or load_profiles()
    config = (
        load_metadata_config(config_path)
        if config_path is not None
        else MetadataConfig()
    )
    detected = source.profile.family if source.profile else None
    selected = profile_override or config.profile or detected
    provenance = "locked-profile" if detected else "none"
    if detected is not None and selected != detected:
        raise ValueError(
            f"checkpoint requires metadata profile {detected!r}; got {selected!r}"
        )
    if selected is not None and selected not in registry.families:
        choices = ", ".join(sorted(registry.families))
        raise ValueError(
            f"unknown metadata profile {selected!r}; choose one of: {choices}"
        )
    has_custom = any(
        value is not None
        for value in (
            config.languages,
            config.datasets,
            config.context_length,
            config.license,
        )
    )
    if (
        detected is None
        and selected is None
        and not has_custom
        and config_path is None
        and interactive
        and sys.stdin.isatty()
        and sys.stderr.isatty()
    ):
        selected = _choose_profile(registry)
        provenance = "interactive" if selected else "none"
    family = registry.families[selected] if selected else None
    languages = (
        config.languages
        if config.languages is not None
        else (family.languages if family else ())
    )
    datasets = (
        config.datasets
        if config.datasets is not None
        else (family.datasets if family else ())
    )
    default_context = (
        source.profile.context_length
        if source.profile is not None
        else (family.context_length if family else None)
    )
    default_license = (
        source.profile.license
        if source.profile is not None
        else (family.license if family else None)
    )
    context_length = (
        config.context_length if config.context_length is not None else default_context
    )
    license_id = config.license if config.license is not None else default_license
    if source.profile is not None and (
        context_length != source.profile.context_length
        or license_id != source.profile.license
    ):
        raise ValueError(
            "recognized checkpoint context and license must match its locked profile"
        )
    if has_custom:
        provenance = "config"
    elif profile_override is not None:
        provenance = "cli"
    elif config.profile is not None:
        provenance = "config-profile"
    elif detected is not None:
        provenance = "locked-profile"
    elif selected is None:
        provenance = "none"
    return ReleaseMetadata(
        profile=selected,
        languages=languages,
        datasets=datasets,
        context_length=context_length,
        license=license_id,
        provenance=provenance,
    )


__all__ = [
    "MetadataConfig",
    "ReleaseMetadata",
    "load_metadata_config",
    "resolve_release_metadata",
]
