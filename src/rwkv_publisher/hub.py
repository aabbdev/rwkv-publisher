from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any

from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationDelete,
    HfApi,
    RepoFile,
    hf_hub_download,
)
from huggingface_hub.errors import RepositoryNotFoundError

from .assets import sha256_file
from .manifest import MANIFEST_NAME, validate_release
from .source import _parse_hub_source

REVISION_PATTERN = re.compile(r"^(?!/)(?!.*\.\.)(?!.*//)[A-Za-z0-9._/-]+(?<!/)$")
REPO_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*/RWKV7-[0-9]+(?:\.[0-9]+)?B-[0-9]{8}$"
)


def _validate_revision(revision: str) -> str:
    if not REVISION_PATTERN.fullmatch(revision):
        raise ValueError(f"unsafe Hub revision: {revision!r}")
    return revision


def _target_repo(release_dir: Path, repo_id: str | None) -> str:
    expected_name = release_dir.name
    repo_id = repo_id or f"BlinkDL/{expected_name}"
    if not REPO_PATTERN.fullmatch(repo_id) or repo_id.split("/", 1)[1] != expected_name:
        raise ValueError(f"repository must be OWNER/{expected_name}; got {repo_id!r}")
    return repo_id


def _remote_metadata_overlay(
    release_dir: Path, repo_id: str, temporary: Path
) -> tuple[dict[str, Path], Path, dict[str, Any]]:
    manifest = json.loads((release_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    default_repo = f"BlinkDL/{release_dir.name}"
    overrides: dict[str, Path] = {}
    for relative in ("README.md",):
        text = (release_dir / relative).read_text(encoding="utf-8")
        path = temporary / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.replace(default_repo, repo_id), encoding="utf-8")
        overrides[relative] = path
        manifest["files"][relative] = {
            **manifest["files"][relative],
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest_path = temporary / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return overrides, manifest_path, manifest


def _expected_remote_files(manifest: dict[str, Any]) -> set[str]:
    return set(manifest["files"]) | {MANIFEST_NAME}


def _verify_unregistered_source(manifest: dict[str, Any]) -> None:
    if manifest.get("profile", {}).get("checkpoint") is not None:
        return
    source = manifest["source"]
    reference = source.get("reference")
    revision = source.get("revision")
    if reference is None or revision is None:
        raise ValueError("unregistered release has no publishable source reference")
    repo_id, filename = _parse_hub_source(reference)
    if filename is None:
        raise ValueError("unregistered source reference has no checkpoint path")
    remote = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename=filename,
            revision=revision,
        )
    ).resolve(strict=True)
    if (
        remote.stat().st_size != source["size_bytes"]
        or sha256_file(remote) != source["sha256"]
    ):
        raise ValueError("unregistered source reference no longer matches the release")


def _verify_remote(
    api: HfApi,
    *,
    repo_id: str,
    revision: str,
    manifest: dict[str, Any],
) -> None:
    expected = _expected_remote_files(manifest)
    entries = {
        entry.path: entry
        for entry in api.list_repo_tree(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            recursive=True,
            expand=True,
        )
        if isinstance(entry, RepoFile)
    }
    if set(entries) != expected:
        raise RuntimeError(
            f"remote file inventory mismatch: missing={sorted(expected - set(entries))}, "
            f"extra={sorted(set(entries) - expected)}"
        )
    for relative, record in manifest["files"].items():
        entry = entries[relative]
        if getattr(entry, "size", None) != record["size_bytes"]:
            raise RuntimeError(f"remote size mismatch: {relative}")
        lfs = getattr(entry, "lfs", None)
        if lfs is not None and getattr(lfs, "sha256", None) != record["sha256"]:
            raise RuntimeError(f"remote LFS SHA-256 mismatch: {relative}")
        if lfs is not None:
            continue
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="model",
                filename=relative,
                revision=revision,
            )
        )
        if sha256_file(downloaded) != record["sha256"]:
            raise RuntimeError(f"remote SHA-256 mismatch: {relative}")
    downloaded_manifest = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename=MANIFEST_NAME,
            revision=revision,
        )
    )
    remote_manifest = json.loads(downloaded_manifest.read_text(encoding="utf-8"))
    if remote_manifest != manifest:
        raise RuntimeError("remote release manifest differs from uploaded manifest")
    if api.model_info(repo_id, revision=revision).sha != revision:
        raise RuntimeError("remote revision did not resolve to the published commit")


def publish_release(
    release_dir: Path,
    *,
    repo_id: str | None = None,
    revision: str = "main",
    private: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    release_dir = release_dir.expanduser().resolve()
    local_manifest = validate_release(release_dir)
    if not local_manifest.get("source", {}).get("reference"):
        raise ValueError(
            "release source provenance is not publishable; rebuild with --source-ref"
        )
    revision = _validate_revision(revision)
    repo_id = _target_repo(release_dir, repo_id)
    total_bytes = sum(item["size_bytes"] for item in local_manifest["files"].values())
    if dry_run:
        return {
            "dry_run": True,
            "release": str(release_dir),
            "repo_id": repo_id,
            "revision": revision,
            "private": private,
            "file_count": len(local_manifest["files"]) + 1,
            "total_bytes": total_bytes,
            "adds": sorted(_expected_remote_files(local_manifest)),
            "deletes": None,
            "deletes_note": "remote deletes require network and are resolved at publish time",
        }

    api = HfApi()
    api.whoami()
    _verify_unregistered_source(local_manifest)
    try:
        existing = api.model_info(repo_id, token=True)
    except RepositoryNotFoundError:
        api.create_repo(
            repo_id=repo_id, repo_type="model", private=private, exist_ok=False
        )
        existing = api.model_info(repo_id, token=True)
    else:
        if bool(existing.private) != bool(private):
            raise ValueError(
                f"existing repository visibility is private={existing.private}, requested private={private}"
            )
    if revision != "main":
        api.create_branch(
            repo_id=repo_id, repo_type="model", branch=revision, exist_ok=True
        )
    parent = api.model_info(repo_id, revision=revision, token=True).sha
    remote_files = set(
        api.list_repo_files(repo_id, repo_type="model", revision=revision, token=True)
    )
    with tempfile.TemporaryDirectory(
        prefix="rwkv-publisher-metadata-"
    ) as temporary_name:
        temporary = Path(temporary_name)
        overrides, manifest_path, remote_manifest = _remote_metadata_overlay(
            release_dir, repo_id, temporary
        )
        expected = _expected_remote_files(remote_manifest)
        operations = []
        for relative in sorted(remote_manifest["files"]):
            path = (
                overrides[relative] if relative in overrides else release_dir / relative
            )
            operations.append(
                CommitOperationAdd(path_in_repo=relative, path_or_fileobj=path)
            )
        operations.append(
            CommitOperationAdd(
                path_in_repo=MANIFEST_NAME, path_or_fileobj=manifest_path
            )
        )
        operations.extend(
            CommitOperationDelete(path_in_repo=relative)
            for relative in sorted(remote_files - expected)
        )
        commit = api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            parent_commit=parent,
            commit_message=f"Publish {release_dir.name}",
            operations=operations,
        )
    validate_release(release_dir)
    commit_sha = commit.oid
    _verify_remote(api, repo_id=repo_id, revision=commit_sha, manifest=remote_manifest)
    return {
        "dry_run": False,
        "release": str(release_dir),
        "repo_id": repo_id,
        "revision": revision,
        "private": private,
        "commit": commit_sha,
        "url": f"https://huggingface.co/{repo_id}/tree/{commit_sha}",
        "file_count": len(expected),
        "total_bytes": total_bytes,
    }


__all__ = ["publish_release"]
