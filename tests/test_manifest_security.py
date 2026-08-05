from __future__ import annotations

import json
from pathlib import Path

import pytest

from rwkv_publisher.manifest import inspect_weights


def test_safetensors_index_rejects_parent_traversal(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    save_file({"weight": torch.zeros(1)}, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 4},
                "weight_map": {"weight": "../outside.safetensors"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsafe shard path"):
        inspect_weights(tmp_path)


def test_unindexed_orphan_safetensors_is_rejected(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    save_file({"weight": torch.zeros(1)}, tmp_path / "model.safetensors")
    save_file({"other": torch.zeros(1)}, tmp_path / "model-extra.safetensors")
    with pytest.raises(ValueError, match="orphaned"):
        inspect_weights(tmp_path)


def test_non_floating_safetensors_is_rejected(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    save_file(
        {"weight": torch.zeros(1, dtype=torch.int64)}, tmp_path / "model.safetensors"
    )
    with pytest.raises(ValueError, match="non-floating"):
        inspect_weights(tmp_path)
