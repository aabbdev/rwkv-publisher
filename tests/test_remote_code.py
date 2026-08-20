from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest

from rwkv_publisher.assets import asset_root
from rwkv_publisher.remote_code import (
    MODEL_CODE_FILENAMES,
    REMOTE_AUTO_MAP,
    REMOTE_CODE_FILES,
    SOURCE_REVISION,
    TRANSFORMERS_MIN_VERSION,
    build_model_code,
    build_remote_code,
    model_code_provenance,
    transform_remote_source,
)

EXPECTED_SOURCE_HASHES = {
    "configuration_rwkv7.py": "6f5b92c5fe7498ad22b0054a2f735a7ca82e7577436f4ad32f0fc27d1e900fdd",
    "modeling_rwkv7.py": "3e8e5af7c4eba0b5de1496aef44773d7ac1bb4d96756e6f55efaf29453d67952",
}


def _sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def test_remote_code_assets_and_provenance_are_pinned() -> None:
    export = build_remote_code()

    assert MODEL_CODE_FILENAMES == REMOTE_CODE_FILES
    assert REMOTE_AUTO_MAP == {
        "AutoConfig": "configuration_rwkv7.Rwkv7Config",
        "AutoModel": "modeling_rwkv7.Rwkv7Model",
        "AutoModelForCausalLM": "modeling_rwkv7.Rwkv7ForCausalLM",
    }
    assert tuple(export.files) == REMOTE_CODE_FILES
    assert build_model_code() == export.files
    assert model_code_provenance() == export.provenance
    assert export.provenance["source_revision"] == SOURCE_REVISION
    assert export.provenance["transformers_min_version"] == TRANSFORMERS_MIN_VERSION
    assert set(export.provenance["sources"]) == set(REMOTE_CODE_FILES)
    for filename, expected_hash in EXPECTED_SOURCE_HASHES.items():
        raw = (asset_root() / "model_code" / filename).read_text(encoding="utf-8")
        item = export.provenance["sources"][filename]
        assert _sha256(raw) == expected_hash == item["source_sha256"]
        assert item["asset_path"] == f"model_code/{filename}"
        assert item["repository_path"].endswith(f"/rwkv7/{filename}")
        assert item["output_sha256"] == _sha256(export.files[filename])


def test_remote_code_changes_only_allowlisted_deep_imports() -> None:
    export = build_remote_code()
    replacements = {
        "from transformers.configuration_utils import PreTrainedConfig": "from ...configuration_utils import PreTrainedConfig",
        "from transformers.utils import auto_docstring": "from ...utils import auto_docstring",
        "from transformers import initialization as init": "from ... import initialization as init",
        "from transformers.cache_utils import Cache, LinearAttentionLayer": "from ...cache_utils import Cache, LinearAttentionLayer",
        "from transformers.generation import GenerationMixin": "from ...generation import GenerationMixin",
        "from transformers.modeling_layers import GradientCheckpointingLayer": "from ...modeling_layers import GradientCheckpointingLayer",
        "from transformers.modeling_utils import PreTrainedModel": "from ...modeling_utils import PreTrainedModel",
        "from transformers.utils import ModelOutput, auto_docstring, can_return_tuple, logging": "from ...utils import ModelOutput, auto_docstring, can_return_tuple, logging",
    }

    for filename, published in export.files.items():
        compile(published, filename, "exec")
        assert "from ..." not in published
        restored = published
        for adapted, original in replacements.items():
            restored = restored.replace(adapted, original)
        raw = (asset_root() / "model_code" / filename).read_text(encoding="utf-8")
        assert restored == raw
    assert (
        "from .configuration_rwkv7 import Rwkv7Config"
        in export.files["modeling_rwkv7.py"]
    )


@pytest.mark.parametrize(
    ("filename", "mutation", "message"),
    [
        (
            "configuration_rwkv7.py",
            lambda source: source.replace(
                "from ...utils import auto_docstring", "from ...utils import renamed"
            ),
            "expected import exactly once",
        ),
        (
            "configuration_rwkv7.py",
            lambda source: source.replace(
                "from ...utils import auto_docstring",
                "from ...utils import auto_docstring\nfrom ....unexpected import value",
            ),
            "unsupported relative import",
        ),
        (
            "modeling_rwkv7.py",
            lambda source: source.replace(
                "from .configuration_rwkv7 import Rwkv7Config", ""
            ),
            "preserve exactly one",
        ),
    ],
)
def test_remote_code_transform_fails_closed(
    filename: str, mutation: Callable[[str], str], message: str
) -> None:
    raw = (asset_root() / "model_code" / filename).read_text(encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        transform_remote_source(mutation(raw), filename)


def test_remote_code_rejects_unknown_source_and_symlink(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported remote-code source"):
        transform_remote_source("", "other.py")

    source_root = tmp_path / "model_code"
    source_root.mkdir()
    for filename in REMOTE_CODE_FILES:
        (source_root / filename).symlink_to(asset_root() / "model_code" / filename)
    with pytest.raises(FileNotFoundError, match="source does not exist"):
        build_remote_code(source_root)
