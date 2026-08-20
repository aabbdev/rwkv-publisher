from __future__ import annotations

import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from tokenizers import Tokenizer

from ._version import VERSION
from .assets import (
    ASSET_SET,
    asset_path,
    asset_root,
    canonical_license_path,
    sha256_file,
    verify_assets,
)
from .conversion import ConversionResult, _expected_shapes
from .encoding import END_TOKEN, build_fast_tokenizer
from .metadata import ReleaseMetadata
from .profiles import load_profiles
from .remote_code import (
    MODEL_CODE_FILENAMES,
    REMOTE_AUTO_MAP,
    build_model_code,
    model_code_provenance,
)
from .runtime_export import build_flat_runtime
from .source import ResolvedSource

MANIFEST_NAME = "release-manifest.json"
MANIFEST_SCHEMA = 6
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "BF16": 2,
    "F16": 2,
    "I16": 2,
    "U16": 2,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}
FLOAT_DTYPES = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
    "F64": "float64",
}


@dataclass(frozen=True)
class WeightFacts:
    files: tuple[dict[str, Any], ...]
    parameter_count: int
    tensor_count: int
    float_dtypes: tuple[str, ...]
    tensor_map_sha256: str
    tensor_shapes: dict[str, tuple[int, ...]]


def _safe_relative(relative: str) -> PurePosixPath:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
        raise ValueError(f"unsafe release path: {relative!r}")
    return pure


def _header(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"invalid safetensors header: {path.name}")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size <= 0 or header_size > path.stat().st_size - 8:
            raise ValueError(f"invalid safetensors header length: {path.name}")
        document = json.loads(handle.read(header_size))
    tensors = {name: item for name, item in document.items() if name != "__metadata__"}
    if not tensors:
        raise ValueError(f"empty safetensors file: {path.name}")
    payload = path.stat().st_size - 8 - header_size
    spans = []
    for name, item in tensors.items():
        dtype = item.get("dtype")
        shape = item.get("shape")
        offsets = item.get("data_offsets")
        if (
            dtype not in DTYPE_BYTES
            or not isinstance(shape, list)
            or not isinstance(offsets, list)
        ):
            raise ValueError(f"invalid tensor metadata: {path.name}:{name}")
        start, end = offsets
        count = 1
        for dimension in shape:
            if not isinstance(dimension, int) or dimension < 0:
                raise ValueError(f"invalid tensor shape: {path.name}:{name}")
            count *= dimension
        if end - start != count * DTYPE_BYTES[dtype] or start < 0 or end > payload:
            raise ValueError(f"invalid tensor span: {path.name}:{name}")
        spans.append((start, end, name))
    for previous, current in zip(sorted(spans), sorted(spans)[1:], strict=False):
        if current[0] < previous[1]:
            raise ValueError(f"overlapping safetensors spans: {path.name}")
    return tensors


def inspect_weights(root: Path) -> WeightFacts:
    index_path = root / "model.safetensors.index.json"
    weight_map: dict[str, str] | None = None
    present_shards = {
        path.name
        for path in root.glob("*.safetensors")
        if path.is_file() and not path.is_symlink()
    }
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("invalid safetensors index")
        names = sorted(set(weight_map.values()))
        if any(
            not isinstance(name, str)
            or PurePosixPath(name).name != name
            or re.fullmatch(r"model-[0-9]{5}-of-[0-9]{5}\.safetensors", name) is None
            for name in names
        ):
            raise ValueError("safetensors index contains an unsafe shard path")
        if set(names) != present_shards:
            raise ValueError("indexed and present safetensors shards differ")
    else:
        names = ["model.safetensors"]
        if present_shards != {"model.safetensors"}:
            raise ValueError("unindexed safetensors shards are ambiguous or orphaned")
    paths = [root / name for name in names]
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("missing or unsafe safetensors shard")
    parameter_count = 0
    tensor_count = 0
    float_dtypes: set[str] = set()
    tensor_digest = hashlib.sha256()
    seen = set()
    tensor_shapes = {}
    files = []
    for path in paths:
        tensors = _header(path)
        files.append(
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        for name, item in sorted(tensors.items()):
            if name in seen:
                raise ValueError(f"duplicate tensor across shards: {name}")
            seen.add(name)
            if weight_map is not None and weight_map.get(name) != path.name:
                raise ValueError(f"safetensors index points to wrong shard: {name}")
            shape = tuple(item["shape"])
            tensor_shapes[name] = shape
            count = 1
            for dimension in shape:
                count *= dimension
            parameter_count += count
            tensor_count += 1
            dtype = item["dtype"]
            if dtype not in FLOAT_DTYPES:
                raise ValueError(f"model contains a non-floating tensor: {name}")
            float_dtypes.add(FLOAT_DTYPES[dtype])
            tensor_digest.update(f"{name}\0{dtype}\0{shape}\n".encode())
    if index_path.is_file():
        index_names = set(
            json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        )
        if index_names != seen:
            raise ValueError(
                "safetensors index tensor map does not match shard headers"
            )
    return WeightFacts(
        files=tuple(files),
        parameter_count=parameter_count,
        tensor_count=tensor_count,
        float_dtypes=tuple(sorted(float_dtypes)),
        tensor_map_sha256=tensor_digest.hexdigest(),
        tensor_shapes=tensor_shapes,
    )


def _role(relative: str) -> str:
    if relative.endswith(".safetensors"):
        return "weights"
    if relative == "README.md":
        return "model_card"
    if relative.startswith("inference/"):
        return "inference"
    if relative in MODEL_CODE_FILENAMES:
        return "model_code"
    if relative in {"config.json", "generation_config.json"}:
        return "model_config"
    if relative in {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}:
        return "tokenizer"
    if relative in {
        "LICENSE",
        "NOTICE",
        ".gitattributes",
        "model.safetensors.index.json",
    }:
        return "metadata"
    raise ValueError(f"unclassified release file: {relative}")


def build_file_inventory(root: Path) -> dict[str, dict[str, Any]]:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.name == MANIFEST_NAME:
            continue
        if path.is_symlink() or (
            path.exists() and not path.is_file() and not path.is_dir()
        ):
            raise ValueError(f"release contains unsafe file: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            _safe_relative(relative)
            files[relative] = {
                "role": _role(relative),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return files


def write_manifest(root: Path, document: dict[str, Any]) -> Path:
    if document.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("invalid release manifest schema")
    document = dict(document)
    document["files"] = build_file_inventory(root)
    path = root / MANIFEST_NAME
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def validate_release(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("release has no regular release-manifest.json")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError(f"release manifest schema must be {MANIFEST_SCHEMA}")
    if set(document) != {
        "schema_version",
        "builder",
        "source",
        "profile",
        "metadata",
        "identity",
        "conversion",
        "model_code",
        "runtime",
        "files",
    }:
        raise ValueError("release manifest has unexpected top-level fields")
    expected_files = document.get("files")
    if not isinstance(expected_files, dict):
        raise TypeError("release manifest files must be an object")
    actual_files = build_file_inventory(root)
    if set(actual_files) != set(expected_files):
        raise ValueError("release file inventory does not match manifest")
    for relative, expected in expected_files.items():
        actual = actual_files[relative]
        if actual["size_bytes"] != expected.get("size_bytes"):
            raise ValueError(f"release file size mismatch: {relative}")
        if actual["sha256"] != expected.get("sha256"):
            raise ValueError(f"release file SHA-256 mismatch: {relative}")
    forbidden = {
        "vocab.json",
        "rwkv_vocab_v20230424.txt",
        "release.toml",
        "rwkv7-conversion.json",
    }
    if forbidden & set(actual_files):
        raise ValueError("release contains forbidden source or project files")
    if (root / "inference/rwkv7_pytorch").exists() or (
        root / "inference/decode"
    ).exists():
        raise ValueError("release contains obsolete nested inference layout")
    if not (root / "inference/kernel.py").is_file():
        raise ValueError("release has no flat inference/kernel.py")
    expected_inference = {
        "generate.py",
        "kernel.py",
        "model_loader.py",
        "requirements.txt",
        "runtime.py",
    }
    actual_inference = {
        path.relative_to(root / "inference").as_posix()
        for path in (root / "inference").rglob("*")
        if path.is_file()
    }
    if actual_inference != expected_inference:
        raise ValueError(
            "release inference tree does not match the flat runtime contract"
        )
    root_python = {path.name for path in root.glob("*.py")}
    if root_python != set(MODEL_CODE_FILENAMES):
        raise ValueError(
            "model root Python files do not match the locked remote-code allowlist: "
            f"{sorted(root_python)}"
        )
    expected_model_code = build_model_code()
    for filename, expected_source in expected_model_code.items():
        if (root / filename).read_text(encoding="utf-8") != expected_source:
            raise ValueError(f"model code does not match locked source: {filename}")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "rwkv7" or config.get("architectures") != [
        "Rwkv7ForCausalLM"
    ]:
        raise ValueError("release config is not native RWKV-7")
    if config.get("auto_map") != REMOTE_AUTO_MAP:
        raise ValueError("release config remote auto_map does not match locked policy")
    if config.get("dtype") not in {"float32", "float16", "bfloat16"}:
        raise ValueError("native release has unsupported dtype")
    generation = json.loads(
        (root / "generation_config.json").read_text(encoding="utf-8")
    )
    if generation != {
        "bos_token_id": 0,
        "eos_token_id": 0,
        "pad_token_id": 0,
        "use_cache": True,
    }:
        raise ValueError("generation config contains unsupported policy")
    tokenizer_config = json.loads(
        (root / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    if tokenizer_config.get("tokenizer_class") != "PreTrainedTokenizerFast" or any(
        key in tokenizer_config for key in ("auto_map", "bos_token", "model_max_length")
    ):
        raise ValueError("tokenizer config is not native and position-neutral")
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    if tokenizer.token_to_id(
        END_TOKEN
    ) != 0 or tokenizer.get_vocab_size() != config.get("vocab_size"):
        raise ValueError("tokenizer does not match native config")
    expected_tokenizer = build_fast_tokenizer(
        asset_path("vocab/rwkv_vocab_v20230424.txt"), int(config["vocab_size"])
    )
    if tokenizer.to_str(pretty=False) != expected_tokenizer.to_str(pretty=False):
        raise ValueError("release tokenizer does not match the locked vocabulary")
    facts = inspect_weights(root)
    expected_shapes = _expected_shapes(config)
    if facts.tensor_shapes != expected_shapes:
        raise ValueError("safetensors tensor keys or shapes do not match config")
    conversion = document.get("conversion", {})
    if conversion.get("serialized_parameter_count") != facts.parameter_count:
        raise ValueError("manifest parameter count does not match weights")
    if conversion.get("tensor_count") != facts.tensor_count:
        raise ValueError("manifest tensor count does not match weights")
    if tuple(sorted(conversion.get("target_float_dtypes", []))) != facts.float_dtypes:
        raise ValueError("manifest dtype does not match weights")
    if conversion.get("tensor_map_sha256") != facts.tensor_map_sha256:
        raise ValueError("manifest tensor map does not match weights")
    if conversion.get("target_float_dtypes") != [config["dtype"]]:
        raise ValueError("config dtype does not match conversion dtype")
    synthesized = conversion.get("synthesized_tensors")
    allowed_synthesized = {
        "rwkv7.blocks.0.att.v0",
        "rwkv7.blocks.0.att.v1",
        "rwkv7.blocks.0.att.v2",
    }
    if not isinstance(synthesized, list) or not set(synthesized) <= allowed_synthesized:
        raise ValueError("manifest has unsupported synthesized tensors")
    synthesized_parameters = sum(
        _shape_size(facts.tensor_shapes[name]) for name in synthesized
    )
    if (
        conversion.get("source_parameter_count") + synthesized_parameters
        != facts.parameter_count
    ):
        raise ValueError("source and synthesized parameter counts do not reconcile")
    identity = document.get("identity", {})
    expected_name = (
        f"RWKV7-{identity.get('parameter_label')}B-{identity.get('release_date')}"
    )
    if root.name != expected_name:
        raise ValueError(f"release directory must be named {expected_name}")
    assets = verify_assets()
    builder = document.get("builder", {})
    if (
        builder.get("version") != VERSION
        or builder.get("asset_set") != ASSET_SET
        or builder.get("assets_sha256") != assets["combined_sha256"]
    ):
        raise ValueError("manifest builder or asset lock does not match this publisher")
    if document.get("model_code") != model_code_provenance():
        raise ValueError("model code provenance does not match locked assets")
    profile_data = document.get("profile", {})
    checkpoint_id = profile_data.get("checkpoint")
    family_id = profile_data.get("family")
    profile = None
    if checkpoint_id is None:
        if family_id is not None:
            raise ValueError("unknown checkpoint must not declare a profile family")
        source = document.get("source", {})
        if (
            source.get("kind") != "local"
            or not isinstance(source.get("filename"), str)
            or PurePosixPath(source["filename"]).name != source["filename"]
            or not source["filename"].lower().endswith(".pth")
            or not re.fullmatch(r"[0-9a-f]{64}", str(source.get("sha256", "")))
            or not isinstance(source.get("size_bytes"), int)
            or source["size_bytes"] <= 0
        ):
            raise ValueError("unregistered release has invalid source provenance")
        reference = source.get("reference")
        revision = source.get("revision")
        if (reference is None) != (revision is None) or (
            revision is not None and not re.fullmatch(r"[0-9a-f]{40}", revision)
        ):
            raise ValueError("unregistered source reference is not immutable")
        if reference is not None and (
            not isinstance(reference, str)
            or re.fullmatch(r"[^/]+/[^/]+/.+\.pth", reference) is None
            or PurePosixPath(reference).name != source["filename"]
        ):
            raise ValueError("unregistered source reference is not canonical")
    else:
        registry = load_profiles()
        try:
            profile = registry.checkpoints[checkpoint_id]
        except KeyError as error:
            raise ValueError("manifest names an unknown checkpoint profile") from error
        family = registry.families[profile.family]
        source = document.get("source", {})
        if (
            family_id != profile.family
            or source.get("kind") not in {"local", "huggingface"}
            or source.get("reference") != profile.source_reference
            or source.get("filename") != profile.filename
            or source.get("sha256") != profile.sha256
            or source.get("size_bytes") != profile.size_bytes
            or source.get("revision") != profile.hub_revision
            or identity.get("parameter_label") != profile.parameter_label
            or identity.get("release_date") != profile.release_date
            or identity.get("training_context_length") != profile.context_length
        ):
            raise ValueError("release facts do not match the locked checkpoint profile")
    metadata_data = document.get("metadata", {})
    if not isinstance(metadata_data, dict) or set(metadata_data) != {
        "profile",
        "languages",
        "datasets",
        "context_length",
        "license",
        "provenance",
    }:
        raise ValueError("release metadata has invalid fields")
    metadata_profile = metadata_data.get("profile")
    languages = _metadata_strings(metadata_data.get("languages"), "languages")
    datasets = _metadata_strings(metadata_data.get("datasets"), "datasets")
    metadata_context = metadata_data.get("context_length")
    if metadata_context is not None and (
        type(metadata_context) is not int or metadata_context <= 0
    ):
        raise ValueError("release metadata context length is invalid")
    metadata_license = metadata_data.get("license")
    if metadata_license is not None and (
        not isinstance(metadata_license, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+-]*", metadata_license) is None
    ):
        raise ValueError("release metadata license is invalid")
    provenance = metadata_data.get("provenance")
    if provenance not in {
        "locked-profile",
        "config-profile",
        "config",
        "cli",
        "interactive",
        "none",
    }:
        raise ValueError("release metadata provenance is invalid")
    registry = load_profiles()
    if metadata_profile is not None and metadata_profile not in registry.families:
        raise ValueError("release metadata names an unknown family profile")
    if profile is not None and metadata_profile != profile.family:
        raise ValueError("release metadata profile does not match the checkpoint")
    if provenance in {"locked-profile", "config-profile", "cli", "interactive"}:
        if metadata_profile is None:
            raise ValueError("profile-derived metadata requires a family profile")
        family = registry.families[metadata_profile]
        if (
            languages != family.languages
            or datasets != family.datasets
            or metadata_context != family.context_length
            or metadata_license != family.license
        ):
            raise ValueError("locked metadata differs from its family profile")
    if provenance == "locked-profile" and profile is None:
        raise ValueError("locked metadata requires a checkpoint profile")
    if provenance == "interactive" and profile is not None:
        raise ValueError("recognized checkpoints cannot use interactive metadata")
    if provenance == "none" and (
        metadata_profile is not None
        or languages
        or datasets
        or metadata_context is not None
        or metadata_license is not None
    ):
        raise ValueError("metadata with no provenance must not contain claims")
    if profile is not None and (
        metadata_context != profile.context_length
        or metadata_license != profile.license
    ):
        raise ValueError("release context or license conflicts with checkpoint profile")
    if identity.get("training_context_length") != metadata_context:
        raise ValueError("release identity context does not match metadata")
    release_metadata = ReleaseMetadata(
        profile=metadata_profile,
        languages=languages,
        datasets=datasets,
        context_length=metadata_context,
        license=metadata_license,
        provenance=provenance,
    )
    canonical_chat = asset_path("templates/chat_template.jinja").read_text(
        encoding="utf-8"
    )
    expected_chat = canonical_chat.replace(
        "bos_token | default('')", "bos_token | default('', true)", 1
    ).replace("tools | default([])", "tools | default([], true)", 1)
    if (root / "chat_template.jinja").read_text(encoding="utf-8") != expected_chat:
        raise ValueError("release chat template does not match the locked asset")
    runtime = document.get("runtime", {})
    expected_runtime = build_flat_runtime(asset_root() / "runtime" / ASSET_SET)
    if runtime != expected_runtime.provenance:
        raise ValueError("runtime provenance does not match the locked runtime")
    for filename, expected_source in expected_runtime.files.items():
        if (root / "inference" / filename).read_text(
            encoding="utf-8"
        ) != expected_source:
            raise ValueError(f"runtime file does not match locked export: {filename}")
    template_root = Path(__file__).resolve().parent / "templates"
    for filename, template_name in (
        ("generate.py", "inference_generate.py.template"),
        ("model_loader.py", "inference_model_loader.py.template"),
        ("requirements.txt", "inference_requirements.txt.template"),
    ):
        expected_text = (template_root / template_name).read_text(encoding="utf-8")
        if (root / "inference" / filename).read_text(encoding="utf-8") != expected_text:
            raise ValueError(f"inference support file differs: {filename}")
    canonical_license = canonical_license_path().read_bytes()
    if (root / "LICENSE").read_bytes() != canonical_license:
        raise ValueError("release license does not match the canonical license")
    card = (root / "README.md").read_text(encoding="utf-8")
    if "trust_remote_code=True" not in card:
        raise ValueError("model card does not document remote-code loading")
    if not card.startswith("---\n") or "\n---\n" not in card[4:]:
        raise ValueError("model card has invalid frontmatter")
    frontmatter = card.split("---\n", 2)[1]
    expected_license = metadata_license or "other"
    if f"license: {expected_license}\n" not in frontmatter:
        raise ValueError("model card license does not match the release")
    if _frontmatter_list(frontmatter, "language", "datasets") != languages:
        raise ValueError("model card languages do not match the locked profile")
    if _frontmatter_list(frontmatter, "datasets", "tags") != datasets:
        raise ValueError("model card datasets do not match the locked profile")
    from .build import _render_card

    source_data = document["source"]
    expected_card = _render_card(
        source=ResolvedSource(
            kind=source_data["kind"],
            local_path=root / source_data["filename"],
            filename=source_data["filename"],
            sha256=source_data["sha256"],
            size_bytes=source_data["size_bytes"],
            reference=source_data.get("reference"),
            revision=source_data.get("revision"),
            profile=profile,
        ),
        conversion=ConversionResult(
            config=config,
            source_float_dtypes=tuple(conversion["source_float_dtypes"]),
            target_float_dtype=conversion["target_float_dtypes"][0],
            explicit_cast=conversion["explicit_cast"],
            source_parameter_count=conversion["source_parameter_count"],
            serialized_parameter_count=conversion["serialized_parameter_count"],
            tensor_count=conversion["tensor_count"],
            synthesized_tensors=tuple(conversion["synthesized_tensors"]),
            weight_bytes=sum(
                item["size_bytes"]
                for relative, item in expected_files.items()
                if relative.endswith(".safetensors")
            ),
            weight_files=(),
        ),
        parameter_label=identity["parameter_label"],
        release_date=identity["release_date"],
        context_length=identity["training_context_length"],
        metadata=release_metadata,
    )
    if card != expected_card:
        raise ValueError("model card does not match the rendered template")
    return document


def _shape_size(shape: tuple[int, ...]) -> int:
    count = 1
    for dimension in shape:
        count *= dimension
    return count


def _frontmatter_list(frontmatter: str, key: str, next_key: str) -> tuple[str, ...]:
    marker = f"{key}:\n"
    end_marker = f"{next_key}:\n"
    if marker not in frontmatter or end_marker not in frontmatter:
        raise ValueError(f"model card frontmatter is missing {key}")
    block = frontmatter.split(marker, 1)[1].split(end_marker, 1)[0].strip("\n")
    if block.strip() == "[]":
        return ()
    values = []
    for line in block.splitlines():
        if not line.startswith("  - "):
            raise ValueError(f"model card frontmatter has invalid {key}")
        try:
            value = json.loads(line.removeprefix("  - "))
        except json.JSONDecodeError as error:
            raise ValueError(f"model card frontmatter has invalid {key}") from error
        if not isinstance(value, str) or not value:
            raise ValueError(f"model card frontmatter has invalid {key}")
        values.append(value)
    return tuple(values)


def _metadata_strings(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"release metadata {field} must be a list of strings")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError(f"release metadata {field} contains duplicates")
    return result


__all__ = [
    "MANIFEST_NAME",
    "MANIFEST_SCHEMA",
    "WeightFacts",
    "build_file_inventory",
    "inspect_weights",
    "validate_release",
    "write_manifest",
]
