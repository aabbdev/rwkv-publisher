from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from huggingface_hub import HfApi, hf_hub_download

from .assets import sha256_file
from .profiles import CheckpointProfile, ProfileRegistry, load_profiles


@dataclass(frozen=True)
class ResolvedSource:
    kind: str
    local_path: Path
    filename: str
    sha256: str
    size_bytes: int
    reference: str | None
    revision: str | None
    profile: CheckpointProfile | None


def _parse_hub_source(source: str) -> tuple[str, str | None]:
    parsed = urlparse(source)
    if parsed.scheme:
        if parsed.scheme != "https" or parsed.hostname not in {
            "huggingface.co",
            "www.huggingface.co",
        }:
            raise ValueError("source URL must use https://huggingface.co")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(
                "source URL must not contain credentials, query, or fragment"
            )
        parts = [unquote(part) for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            raise ValueError("source URL must identify a Hugging Face repository")
        repo_id = "/".join(parts[:2])
        if len(parts) == 2:
            return repo_id, None
        if len(parts) >= 5 and parts[2] in {"blob", "resolve"}:
            return repo_id, "/".join(parts[4:])
        raise ValueError("source URL must identify one checkpoint file")
    parts = [part for part in source.split("/") if part]
    if len(parts) < 2:
        raise ValueError("source must be a local .pth or OWNER/REPOSITORY reference")
    return "/".join(parts[:2]), "/".join(parts[2:]) or None


def _profile_for_hub(
    registry: ProfileRegistry, repo_id: str, filename: str | None
) -> CheckpointProfile:
    matches = registry.for_repo(repo_id)
    if filename is not None:
        matches = tuple(profile for profile in matches if profile.filename == filename)
    if len(matches) != 1:
        choices = [profile.source_reference for profile in registry.for_repo(repo_id)]
        if not choices:
            raise ValueError(f"unregistered Hub checkpoint source: {repo_id}")
        raise ValueError(f"source is ambiguous; choose one of: {choices}")
    return matches[0]


def _verify_profile_file(path: Path, profile: CheckpointProfile) -> tuple[str, int]:
    size = path.stat().st_size
    if size != profile.size_bytes:
        raise ValueError(
            f"checkpoint size mismatch for {profile.filename}: expected={profile.size_bytes}, actual={size}"
        )
    digest = sha256_file(path)
    if digest != profile.sha256:
        raise ValueError(
            f"checkpoint SHA-256 mismatch for {profile.filename}: expected={profile.sha256}, actual={digest}"
        )
    return digest, size


def _resolve_source_reference(
    reference: str,
    *,
    local_path: Path,
    local_sha256: str,
    local_size: int,
    offline: bool,
) -> tuple[str, str]:
    repo_id, filename = _parse_hub_source(reference)
    if filename is None or Path(filename).name != local_path.name:
        raise ValueError("source-ref filename must match the local checkpoint")
    if offline:
        raise ValueError(
            "--source-ref for an unregistered checkpoint requires network access"
        )
    try:
        revision = HfApi().model_info(repo_id).sha
        if not isinstance(revision, str) or len(revision) != 40:
            raise ValueError(
                "source-ref repository did not resolve to an immutable commit"
            )
        remote_link = Path(
            hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
        )
        remote_path = remote_link.resolve(strict=True)
    except Exception as error:
        raise RuntimeError(
            f"could not resolve source-ref at an immutable revision: {reference}"
        ) from error
    if (
        not remote_path.is_file()
        or remote_path.stat().st_size != local_size
        or sha256_file(remote_path) != local_sha256
    ):
        raise ValueError("source-ref file does not match the local checkpoint")
    return f"{repo_id}/{filename}", revision


def resolve_source(
    source: str | Path,
    *,
    source_ref: str | None = None,
    offline: bool = False,
    registry: ProfileRegistry | None = None,
) -> ResolvedSource:
    registry = registry or load_profiles()
    local = Path(source).expanduser()
    if local.exists() or isinstance(source, Path):
        if local.is_symlink() or not local.is_file() or local.suffix.lower() != ".pth":
            raise ValueError(
                f"local source must be a regular non-symlinked .pth: {local}"
            )
        local = local.resolve()
        size = local.stat().st_size
        digest = sha256_file(local)
        profile = registry.by_sha256(digest)
        filename_profile = registry.by_filename(local.name)
        if profile is None and filename_profile is not None:
            raise ValueError(
                f"known checkpoint filename has unexpected SHA-256: {local.name}"
            )
        reference = profile.source_reference if profile else None
        revision = profile.hub_revision if profile else None
        if profile is not None and source_ref is not None:
            repo_id, filename = _parse_hub_source(source_ref)
            if filename != profile.filename or repo_id != profile.hub_repo:
                raise ValueError(
                    "source-ref does not match the registered checkpoint profile"
                )
        elif profile is None and source_ref is not None:
            reference, revision = _resolve_source_reference(
                source_ref,
                local_path=local,
                local_sha256=digest,
                local_size=size,
                offline=offline,
            )
        return ResolvedSource(
            kind="local",
            local_path=local,
            filename=profile.filename if profile else local.name,
            sha256=digest,
            size_bytes=size,
            reference=reference,
            revision=revision,
            profile=profile,
        )

    repo_id, filename = _parse_hub_source(str(source))
    profile = _profile_for_hub(registry, repo_id, filename)
    try:
        downloaded_link = Path(
            hf_hub_download(
                repo_id=profile.hub_repo,
                filename=profile.filename,
                revision=profile.hub_revision,
                local_files_only=offline,
            )
        )
    except Exception as error:
        mode = "offline cache" if offline else "Hugging Face Hub"
        raise RuntimeError(
            f"could not resolve checkpoint from {mode}: {profile.source_reference}"
        ) from error
    try:
        downloaded = downloaded_link.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(
            f"cached checkpoint does not resolve to a regular file: {downloaded_link}"
        ) from error
    if not downloaded.is_file():
        raise RuntimeError(f"cached checkpoint is not a regular file: {downloaded}")
    digest, size = _verify_profile_file(downloaded, profile)
    return ResolvedSource(
        kind="huggingface",
        local_path=downloaded,
        filename=profile.filename,
        sha256=digest,
        size_bytes=size,
        reference=profile.source_reference,
        revision=profile.hub_revision,
        profile=profile,
    )


__all__ = ["ResolvedSource", "resolve_source"]
