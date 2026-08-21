from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .assets import asset_root

REMOTE_CODE_FORMAT_VERSION = 2
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
SFT_COMPATIBILITY_PATCHES = (
    "layer-zero-value-residual-buffers",
    "trainer-past-key-values-placeholder",
    "trl-position-ids-packing-boundaries",
    "labels-disable-recurrent-cache",
)

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
_SOURCE_PATCHES = {
    "modeling_rwkv7.py": (
        (
            """        self.v1 = nn.Parameter(torch.zeros(C, config.v_low_rank_dim))
        self.v2 = nn.Parameter(torch.zeros(config.v_low_rank_dim, C))
        self.v0 = nn.Parameter(torch.zeros(1, 1, C))
""",
            """        if layer_id == 0:
            self.register_buffer("v1", torch.zeros(C, config.v_low_rank_dim))
            self.register_buffer("v2", torch.zeros(config.v_low_rank_dim, C))
            self.register_buffer("v0", torch.zeros(1, 1, C))
        else:
            self.v1 = nn.Parameter(torch.zeros(C, config.v_low_rank_dim))
            self.v2 = nn.Parameter(torch.zeros(config.v_low_rank_dim, C))
            self.v0 = nn.Parameter(torch.zeros(1, 1, C))
""",
        ),
        (
            """    last_hidden_state: torch.FloatTensor | None = None
    state: Rwkv7Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
""",
            """    last_hidden_state: torch.FloatTensor | None = None
    state: Rwkv7Cache | None = None
    past_key_values: None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
""",
        ),
        (
            """@auto_docstring
class Rwkv7Model(Rwkv7PreTrainedModel):
""",
            """def _packed_boundaries_from_position_ids(
    position_ids: torch.LongTensor | None, seq_len: int
) -> torch.LongTensor | None:
    if position_ids is None or position_ids.ndim != 2 or position_ids.shape[0] != 1:
        return None
    starts = torch.nonzero(position_ids[0] == 0, as_tuple=False).flatten()
    if starts.numel() <= 1:
        return None
    if starts[0].item() != 0:
        raise ValueError("packed position_ids must start at zero")
    return torch.cat((starts, starts.new_tensor([seq_len])))


@auto_docstring
class Rwkv7Model(Rwkv7PreTrainedModel):
""",
        ),
        (
            """        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        state: Rwkv7Cache | None = None,
        cu_seq_lens: torch.LongTensor | None = None,
""",
            """        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        state: Rwkv7Cache | None = None,
        cu_seq_lens: torch.LongTensor | None = None,
""",
        ),
        (
            """        if inputs_embeds is None:
            inputs_embeds = self.emb(input_ids)

        if use_cache and state is None:
""",
            """        if inputs_embeds is None:
            inputs_embeds = self.emb(input_ids)

        if cu_seq_lens is None and attention_mask is None:
            cu_seq_lens = _packed_boundaries_from_position_ids(
                position_ids, inputs_embeds.shape[1]
            )

        if use_cache and state is None:
""",
        ),
        (
            """        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        state: Rwkv7Cache | None = None,
        labels: torch.LongTensor | None = None,
""",
            """        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        state: Rwkv7Cache | None = None,
        labels: torch.LongTensor | None = None,
""",
        ),
        (
            """        outputs = self.rwkv7(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
""",
            """        if labels is not None and use_cache is None:
            use_cache = False
        outputs = self.rwkv7(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
""",
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
    for original, replacement in _SOURCE_PATCHES.get(filename, ()):
        if transformed.count(original) != 1:
            raise ValueError(
                f"{filename} must contain patch target exactly once: {original.splitlines()[0]}"
            )
        if transformed.count(replacement) != 0:
            raise ValueError(
                f"{filename} unexpectedly already contains SFT compatibility patch"
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
        "patches": list(SFT_COMPATIBILITY_PATCHES),
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
    "SFT_COMPATIBILITY_PATCHES",
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
