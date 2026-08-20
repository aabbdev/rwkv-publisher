from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .assets import asset_root

REMOTE_CODE_FORMAT_VERSION = 1
REMOTE_CODE_FILES = ("configuration_rwkv7.py", "modeling_rwkv7.py")
MODEL_CODE_FILENAMES = REMOTE_CODE_FILES
REMOTE_AUTO_MAP = {
    "AutoConfig": "configuration_rwkv7.Rwkv7Config",
    "AutoModel": "modeling_rwkv7.Rwkv7Model",
    "AutoModelForCausalLM": "modeling_rwkv7.Rwkv7ForCausalLM",
}
SOURCE_REPOSITORY = "https://github.com/huggingface/transformers.git"
SOURCE_REVISION = "4ad9ed0747ed6ba75c787e8f9040dcd64b166ee2"
SOURCE_DIRECTORY = "src/transformers/models/rwkv7"
TRANSFORMERS_MIN_VERSION = "5.15"

_IMPORT_REPLACEMENTS = {
    "configuration_rwkv7.py": (
        (
            "from ...configuration_utils import PreTrainedConfig",
            "from transformers.configuration_utils import PreTrainedConfig",
        ),
        (
            "from ...utils import auto_docstring",
            "from transformers.utils import auto_docstring",
        ),
    ),
    "modeling_rwkv7.py": (
        (
            "from ... import initialization as init",
            "from transformers import initialization as init",
        ),
        (
            "from ...cache_utils import Cache, LinearAttentionLayer",
            "from transformers.cache_utils import Cache, LinearAttentionLayer",
        ),
        (
            "from ...generation import GenerationMixin",
            "from transformers.generation import GenerationMixin",
        ),
        (
            "from ...modeling_layers import GradientCheckpointingLayer",
            "from transformers.modeling_layers import GradientCheckpointingLayer",
        ),
        (
            "from ...modeling_utils import PreTrainedModel",
            "from transformers.modeling_utils import PreTrainedModel",
        ),
        (
            "from ...utils import ModelOutput, auto_docstring, can_return_tuple, logging",
            "from transformers.utils import ModelOutput, auto_docstring, can_return_tuple, logging",
        ),
    ),
}
_LOCAL_CONFIGURATION_IMPORT = "from .configuration_rwkv7 import Rwkv7Config"


@dataclass(frozen=True)
class RemoteCodeExport:
    files: dict[str, str]
    provenance: dict[str, Any]


def _sha256_text(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _validate_relative_imports(source: str, filename: str) -> None:
    tree = ast.parse(source, filename=filename)
    local_configuration_imports = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level == 0:
            continue
        if (
            filename == "modeling_rwkv7.py"
            and node.level == 1
            and node.module == "configuration_rwkv7"
            and [(item.name, item.asname) for item in node.names]
            == [("Rwkv7Config", None)]
        ):
            local_configuration_imports += 1
            continue
        raise ValueError(
            f"{filename} contains unsupported relative import from {'.' * node.level}{node.module or ''}"
        )
    expected = 1 if filename == "modeling_rwkv7.py" else 0
    if local_configuration_imports != expected:
        raise ValueError(
            f"{filename} must contain the local configuration import exactly {expected} time(s)"
        )


def transform_remote_source(source: str, filename: str) -> str:
    replacements = _IMPORT_REPLACEMENTS.get(filename)
    if replacements is None:
        raise ValueError(f"unsupported remote-code source: {filename}")
    if (
        filename == "modeling_rwkv7.py"
        and source.count(_LOCAL_CONFIGURATION_IMPORT) != 1
    ):
        raise ValueError(
            "modeling_rwkv7.py must preserve exactly one local configuration import"
        )
    transformed = source
    for original, replacement in replacements:
        if source.count(original) != 1:
            raise ValueError(
                f"{filename} must contain expected import exactly once: {original}"
            )
        if source.count(replacement) != 0:
            raise ValueError(
                f"{filename} unexpectedly already contains adapted import: {replacement}"
            )
        transformed = transformed.replace(original, replacement)
    _validate_relative_imports(transformed, filename)
    return transformed


def build_remote_code(source_root: Path | None = None) -> RemoteCodeExport:
    root = source_root if source_root is not None else asset_root() / "model_code"
    files: dict[str, str] = {}
    sources: dict[str, dict[str, str]] = {}
    for filename in REMOTE_CODE_FILES:
        path = root / filename
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"remote-code source does not exist: {path}")
        source = path.read_text(encoding="utf-8")
        transformed = transform_remote_source(source, filename)
        files[filename] = transformed
        sources[filename] = {
            "asset_path": f"model_code/{filename}",
            "repository_path": f"{SOURCE_DIRECTORY}/{filename}",
            "source_sha256": _sha256_text(source),
            "output_sha256": _sha256_text(transformed),
        }
    provenance: dict[str, Any] = {
        "format_version": REMOTE_CODE_FORMAT_VERSION,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "transformers_min_version": TRANSFORMERS_MIN_VERSION,
        "sources": sources,
    }
    return RemoteCodeExport(files=files, provenance=provenance)


def build_model_code(source_root: Path | None = None) -> dict[str, str]:
    return build_remote_code(source_root).files


def model_code_provenance(source_root: Path | None = None) -> dict[str, Any]:
    return build_remote_code(source_root).provenance


__all__ = [
    "MODEL_CODE_FILENAMES",
    "REMOTE_AUTO_MAP",
    "REMOTE_CODE_FILES",
    "REMOTE_CODE_FORMAT_VERSION",
    "SOURCE_DIRECTORY",
    "SOURCE_REPOSITORY",
    "SOURCE_REVISION",
    "TRANSFORMERS_MIN_VERSION",
    "RemoteCodeExport",
    "build_model_code",
    "build_remote_code",
    "model_code_provenance",
    "transform_remote_source",
]
