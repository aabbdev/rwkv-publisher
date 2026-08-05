from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .assets import asset_path


@dataclass(frozen=True)
class FamilyProfile:
    id: str
    languages: tuple[str, ...]
    datasets: tuple[str, ...]
    chat_template: str


@dataclass(frozen=True)
class CheckpointProfile:
    id: str
    family: str
    filename: str
    hub_repo: str
    hub_revision: str
    sha256: str
    size_bytes: int
    parameter_label: str
    release_date: str
    context_length: int
    license: str

    @property
    def source_reference(self) -> str:
        return f"{self.hub_repo}/{self.filename}"

    @property
    def model_name(self) -> str:
        return f"RWKV7-{self.parameter_label}B-{self.release_date}"


@dataclass(frozen=True)
class ProfileRegistry:
    families: dict[str, FamilyProfile]
    checkpoints: dict[str, CheckpointProfile]

    def by_sha256(self, digest: str) -> CheckpointProfile | None:
        matches = [
            profile for profile in self.checkpoints.values() if profile.sha256 == digest
        ]
        if len(matches) > 1:
            raise ValueError(f"duplicate checkpoint SHA-256 in profiles: {digest}")
        return matches[0] if matches else None

    def by_filename(self, filename: str) -> CheckpointProfile | None:
        matches = [
            profile
            for profile in self.checkpoints.values()
            if profile.filename == filename
        ]
        if len(matches) > 1:
            raise ValueError(f"ambiguous checkpoint filename in profiles: {filename}")
        return matches[0] if matches else None

    def for_repo(self, repo_id: str) -> tuple[CheckpointProfile, ...]:
        return tuple(
            profile
            for profile in self.checkpoints.values()
            if profile.hub_repo == repo_id
        )


def _strings(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise TypeError(f"profile {field} must be a list of strings")
    return tuple(value)


def load_profiles() -> ProfileRegistry:
    document = json.loads(asset_path("profiles.json").read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("unsupported profile schema")
    families = {
        name: FamilyProfile(
            id=name,
            languages=_strings(item.get("languages"), f"families.{name}.languages"),
            datasets=_strings(item.get("datasets"), f"families.{name}.datasets"),
            chat_template=str(item.get("chat_template")),
        )
        for name, item in document["families"].items()
    }
    checkpoints = {}
    for name, item in document["checkpoints"].items():
        profile = CheckpointProfile(id=name, **item)
        if profile.family not in families:
            raise ValueError(f"unknown profile family: {profile.family}")
        if len(profile.sha256) != 64 or len(profile.hub_revision) != 40:
            raise ValueError(
                f"invalid immutable identity in checkpoint profile: {name}"
            )
        checkpoints[name] = profile
    return ProfileRegistry(families=families, checkpoints=checkpoints)


__all__ = ["CheckpointProfile", "FamilyProfile", "ProfileRegistry", "load_profiles"]
