from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rwkv_publisher import hub
from rwkv_publisher.cli import main


def _release(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    release = tmp_path / "RWKV7-0.1B-20241210"
    release.mkdir()
    readme = release / "README.md"
    config = release / "config.json"
    readme.write_text("# BlinkDL/RWKV7-0.1B-20241210\n", encoding="utf-8")
    config.write_text("{}\n", encoding="utf-8")
    manifest = {
        "schema_version": 3,
        "profile": {"checkpoint": "test"},
        "source": {"reference": "BlinkDL/source/model.pth"},
        "files": {
            "README.md": {
                "size_bytes": readme.stat().st_size,
                "sha256": hub.sha256_file(readme),
            },
            "config.json": {
                "size_bytes": config.stat().st_size,
                "sha256": hub.sha256_file(config),
            },
        },
    }
    (release / hub.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return release, manifest


def test_publish_dry_run_has_no_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, manifest = _release(tmp_path)
    monkeypatch.setattr(hub, "validate_release", lambda _: manifest)
    monkeypatch.setattr(
        hub, "HfApi", lambda: (_ for _ in ()).throw(AssertionError("network used"))
    )

    result = hub.publish_release(
        release, repo_id="aabbdev/RWKV7-0.1B-20241210", private=True, dry_run=True
    )

    assert result["dry_run"] is True
    assert result["repo_id"] == "aabbdev/RWKV7-0.1B-20241210"
    assert result["file_count"] == 3


def test_publish_rejects_destination_with_wrong_basename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, manifest = _release(tmp_path)
    monkeypatch.setattr(hub, "validate_release", lambda _: manifest)
    with pytest.raises(ValueError, match="repository must be"):
        hub.publish_release(release, repo_id="aabbdev/different", dry_run=True)


class FakeApi:
    def __init__(self) -> None:
        self.commit_kwargs: dict[str, Any] | None = None

    def whoami(self) -> dict[str, str]:
        return {"name": "aabbdev"}

    def model_info(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(private=False, sha="parent-sha")

    def list_repo_files(self, *args: Any, **kwargs: Any) -> list[str]:
        return ["stale.txt"]

    def create_commit(self, **kwargs: Any) -> SimpleNamespace:
        self.commit_kwargs = kwargs
        return SimpleNamespace(oid="immutable-commit")


def test_publish_uses_parent_locked_atomic_commit_and_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, manifest = _release(tmp_path)
    validations: list[Path] = []

    def validate(path: Path) -> dict[str, Any]:
        validations.append(path)
        return manifest

    api = FakeApi()
    verified: dict[str, Any] = {}
    monkeypatch.setattr(hub, "validate_release", validate)
    monkeypatch.setattr(hub, "HfApi", lambda: api)
    monkeypatch.setattr(
        hub,
        "_verify_remote",
        lambda *args, **kwargs: verified.update(kwargs),
    )

    result = hub.publish_release(release, repo_id="aabbdev/RWKV7-0.1B-20241210")

    assert result["commit"] == "immutable-commit"
    assert len(validations) == 2
    assert api.commit_kwargs is not None
    assert api.commit_kwargs["parent_commit"] == "parent-sha"
    paths = [operation.path_in_repo for operation in api.commit_kwargs["operations"]]
    assert paths == [
        "README.md",
        "config.json",
        hub.MANIFEST_NAME,
        "stale.txt",
    ]
    assert verified["revision"] == "immutable-commit"
    assert (
        verified["manifest"]["files"]["README.md"]["sha256"]
        != manifest["files"]["README.md"]["sha256"]
    )


def test_publish_cli_is_immediate_unless_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    release, _ = _release(tmp_path)
    captured: dict[str, Any] = {}

    def fake_publish(path: Path, **kwargs: Any) -> dict[str, Any]:
        captured.update(path=path, **kwargs)
        return {"dry_run": False, "url": "https://example.test/commit"}

    monkeypatch.setattr("rwkv_publisher.cli.publish_release", fake_publish)
    main(
        [
            "publish",
            str(release),
            "--repo",
            "aabbdev/RWKV7-0.1B-20241210",
            "--json",
        ]
    )
    assert captured["dry_run"] is False
    assert json.loads(capsys.readouterr().out)["url"].endswith("/commit")
