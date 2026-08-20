from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .assets import asset_path
from .encoding import END_TOKEN, build_fast_tokenizer
from .remote_code import REMOTE_AUTO_MAP

DTYPES = {"float32", "float16", "bfloat16"}
SIZE_UNITS = {
    "B": 1,
    "KB": 1_000,
    "MB": 1_000_000,
    "GB": 1_000_000_000,
    "KIB": 1 << 10,
    "MIB": 1 << 20,
    "GIB": 1 << 30,
}


@dataclass(frozen=True)
class ConversionResult:
    config: dict[str, Any]
    source_float_dtypes: tuple[str, ...]
    target_float_dtype: str
    explicit_cast: bool
    source_parameter_count: int
    serialized_parameter_count: int
    tensor_count: int
    synthesized_tensors: tuple[str, ...]
    weight_bytes: int
    weight_files: tuple[dict[str, Any], ...]


def _size_bytes(value: str | int) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    match = re.fullmatch(
        r"([1-9][0-9]*(?:\.[0-9]+)?)\s*([KMG]?I?B)", str(value).upper()
    )
    if match is None:
        raise ValueError(f"invalid shard size: {value!r}")
    return int(float(match.group(1)) * SIZE_UNITS[match.group(2)])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    import torch  # pyright: ignore[reportMissingImports]

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (RuntimeError, ValueError):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must contain a tensor dictionary")
    for wrapper in ("state_dict", "model", "weights"):
        nested = checkpoint.get(wrapper)
        if isinstance(nested, dict):
            checkpoint = nested
            break
    if not checkpoint or any(not isinstance(key, str) for key in checkpoint):
        raise TypeError("checkpoint must contain string tensor names")
    if any(not isinstance(value, torch.Tensor) for value in checkpoint.values()):
        raise TypeError("checkpoint contains non-tensor values")
    return checkpoint


def _native_key(key: str) -> str:
    if key == "emb.weight":
        return "rwkv7.emb.weight"
    if key == "head.weight":
        return "head.weight"
    if key.startswith(("ln_out.", "blocks.")):
        return f"rwkv7.{key}"
    raise KeyError(f"unrecognized native checkpoint key: {key!r}")


def _convert_native(state_dict: dict[str, Any]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for key, tensor in state_dict.items():
        target = _native_key(key)
        if target in converted:
            raise ValueError(f"duplicate converted tensor key: {target}")
        converted[target] = tensor
    return converted


def _shape(tensor: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor.shape)


def _source_float_dtypes(state_dict: dict[str, Any]) -> tuple[str, ...]:
    non_floating = sorted(
        name for name, tensor in state_dict.items() if not tensor.is_floating_point()
    )
    if non_floating:
        raise ValueError(
            f"checkpoint contains unsupported non-floating tensors: {non_floating}"
        )
    names = sorted(
        {
            str(tensor.dtype).removeprefix("torch.")
            for tensor in state_dict.values()
            if tensor.is_floating_point()
        }
    )
    if not names:
        raise ValueError("checkpoint contains no floating tensors")
    return tuple(names)


def _target_dtype(source_dtypes: tuple[str, ...], requested: str) -> tuple[str, bool]:
    if requested == "preserve":
        if len(source_dtypes) != 1 or source_dtypes[0] not in DTYPES:
            raise ValueError(
                f"cannot preserve mixed or unsupported source dtypes {source_dtypes}; choose --dtype explicitly"
            )
        return source_dtypes[0], False
    if requested not in DTYPES:
        raise ValueError(
            f"dtype must be preserve or one of {sorted(DTYPES)}, got {requested!r}"
        )
    return requested, source_dtypes != (requested,)


def _expected_shapes(config: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    vocab = int(config["vocab_size"])
    hidden = int(config["hidden_size"])
    heads = int(config["num_heads"])
    head_dim = int(config["head_dim"])
    intermediate = int(config["intermediate_size"])
    ranks = {
        "w": int(config["decay_low_rank_dim"]),
        "a": int(config["a_low_rank_dim"]),
        "v": int(config["v_low_rank_dim"]),
        "g": int(config["gate_low_rank_dim"]),
    }
    expected = {
        "rwkv7.emb.weight": (vocab, hidden),
        "rwkv7.ln_out.weight": (hidden,),
        "rwkv7.ln_out.bias": (hidden,),
        "head.weight": (vocab, hidden),
    }
    for layer in range(int(config["num_hidden_layers"])):
        block = f"rwkv7.blocks.{layer}"
        if layer == 0:
            expected[f"{block}.ln0.weight"] = (hidden,)
            expected[f"{block}.ln0.bias"] = (hidden,)
        for norm in ("ln1", "ln2"):
            expected[f"{block}.{norm}.weight"] = (hidden,)
            expected[f"{block}.{norm}.bias"] = (hidden,)
        attention = f"{block}.att"
        for name in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"):
            expected[f"{attention}.{name}"] = (1, 1, hidden)
        for name in ("w", "a", "v"):
            rank = ranks[name]
            expected[f"{attention}.{name}0"] = (1, 1, hidden)
            expected[f"{attention}.{name}1"] = (hidden, rank)
            expected[f"{attention}.{name}2"] = (rank, hidden)
        expected[f"{attention}.g1"] = (hidden, ranks["g"])
        expected[f"{attention}.g2"] = (ranks["g"], hidden)
        expected[f"{attention}.k_k"] = (1, 1, hidden)
        expected[f"{attention}.k_a"] = (1, 1, hidden)
        expected[f"{attention}.r_k"] = (heads, head_dim)
        for projection in ("receptance", "key", "value", "output"):
            expected[f"{attention}.{projection}.weight"] = (hidden, hidden)
        expected[f"{attention}.ln_x.weight"] = (hidden,)
        expected[f"{attention}.ln_x.bias"] = (hidden,)
        feed_forward = f"{block}.ffn"
        expected[f"{feed_forward}.x_k"] = (1, 1, hidden)
        expected[f"{feed_forward}.key.weight"] = (intermediate, hidden)
        expected[f"{feed_forward}.value.weight"] = (hidden, intermediate)
    return expected


def _infer_config(state_dict: dict[str, Any], dtype: str) -> dict[str, Any]:
    required = {
        "rwkv7.emb.weight",
        "head.weight",
        "rwkv7.ln_out.weight",
        "rwkv7.ln_out.bias",
        "rwkv7.blocks.0.att.r_k",
        "rwkv7.blocks.0.att.w1",
        "rwkv7.blocks.0.att.a1",
        "rwkv7.blocks.0.att.g1",
        "rwkv7.blocks.0.ffn.key.weight",
    }
    missing = sorted(required - set(state_dict))
    if missing:
        raise ValueError(f"checkpoint is missing required tensors: {missing}")
    embedding_shape = _shape(state_dict["rwkv7.emb.weight"])
    if len(embedding_shape) != 2:
        raise ValueError("embedding weight must be rank two")
    vocab_size, hidden_size = embedding_shape
    if _shape(state_dict["head.weight"]) != embedding_shape:
        raise ValueError("head and embedding shapes must match")
    if _shape(state_dict["rwkv7.ln_out.weight"]) != (hidden_size,) or _shape(
        state_dict["rwkv7.ln_out.bias"]
    ) != (hidden_size,):
        raise ValueError("output layer norm does not match hidden size")

    layer_ids = sorted(
        {
            int(match.group(1))
            for key in state_dict
            if (match := re.match(r"rwkv7\.blocks\.(\d+)\.", key)) is not None
        }
    )
    if not layer_ids or layer_ids != list(range(layer_ids[-1] + 1)):
        raise ValueError(f"checkpoint layer indices are not contiguous: {layer_ids}")
    num_layers = len(layer_ids)
    heads, head_dim = _shape(state_dict["rwkv7.blocks.0.att.r_k"])
    if heads * head_dim != hidden_size:
        raise ValueError("attention heads do not match hidden size")
    v_rank_key = next(
        (
            f"rwkv7.blocks.{layer}.att.v1"
            for layer in layer_ids
            if f"rwkv7.blocks.{layer}.att.v1" in state_dict
        ),
        None,
    )
    if v_rank_key is None:
        raise ValueError("checkpoint has no value residual rank")
    config = {
        "architectures": ["Rwkv7ForCausalLM"],
        "auto_map": dict(REMOTE_AUTO_MAP),
        "model_type": "rwkv7",
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "num_heads": heads,
        "head_dim": head_dim,
        "intermediate_size": _shape(state_dict["rwkv7.blocks.0.ffn.key.weight"])[0],
        "decay_low_rank_dim": _shape(state_dict["rwkv7.blocks.0.att.w1"])[1],
        "a_low_rank_dim": _shape(state_dict["rwkv7.blocks.0.att.a1"])[1],
        "v_low_rank_dim": _shape(state_dict[v_rank_key])[1],
        "gate_low_rank_dim": _shape(state_dict["rwkv7.blocks.0.att.g1"])[1],
        "norm_eps": 1e-5,
        "norm_bias": True,
        "wkv_state_dtype": "float32",
        "wkv_implementation": "eager",
        "use_cache": True,
        "tie_word_embeddings": False,
        "bos_token_id": 0,
        "eos_token_id": 0,
        "pad_token_id": 0,
        "dtype": dtype,
    }
    reference = state_dict["rwkv7.emb.weight"]
    state_dict.setdefault(
        "rwkv7.blocks.0.att.v0", reference.new_zeros((1, 1, hidden_size))
    )
    state_dict.setdefault(
        "rwkv7.blocks.0.att.v1",
        reference.new_zeros((hidden_size, int(config["v_low_rank_dim"]))),
    )
    state_dict.setdefault(
        "rwkv7.blocks.0.att.v2",
        reference.new_zeros((int(config["v_low_rank_dim"]), hidden_size)),
    )
    expected = _expected_shapes(config)
    missing = sorted(set(expected) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"checkpoint tensor keys mismatch: missing={missing}, unexpected={unexpected}"
        )
    wrong_shapes = {
        key: (_shape(state_dict[key]), expected[key])
        for key in sorted(expected)
        if _shape(state_dict[key]) != expected[key]
    }
    if wrong_shapes:
        raise ValueError(f"checkpoint tensor shapes mismatch: {wrong_shapes}")
    return config


def _save_sharded_safetensors(
    state_dict: dict[str, Any],
    output_dir: Path,
    max_shard_size: str | int,
    dtype: str,
) -> tuple[list[dict[str, Any]], int]:
    import torch  # pyright: ignore[reportMissingImports]
    from safetensors.torch import save_file

    maximum = _size_bytes(max_shard_size)
    target_dtype = getattr(torch, dtype)
    target_element_size = torch.empty((), dtype=target_dtype).element_size()
    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    total_size = 0
    for name, tensor in state_dict.items():
        element_size = (
            target_element_size if tensor.is_floating_point() else tensor.element_size()
        )
        tensor_size = tensor.numel() * element_size
        if tensor_size > maximum:
            raise ValueError(
                f"tensor {name!r} requires {tensor_size} bytes, larger than shard limit {maximum}"
            )
        total_size += tensor_size
        if current and current_size + tensor_size > maximum:
            shards.append(current)
            current = []
            current_size = 0
        current.append(name)
        current_size += tensor_size
    if current:
        shards.append(current)

    sharded = len(shards) > 1
    weight_map: dict[str, str] = {}
    files: list[dict[str, Any]] = []
    for index, names in enumerate(shards, start=1):
        filename = (
            f"model-{index:05d}-of-{len(shards):05d}.safetensors"
            if sharded
            else "model.safetensors"
        )
        tensors = {}
        for name in names:
            tensor = state_dict[name].detach()
            if tensor.is_floating_point():
                tensor = tensor.to(dtype=target_dtype)
            tensors[name] = tensor.contiguous().clone()
        save_file(tensors, output_dir / filename, metadata={"format": "pt"})
        size = (output_dir / filename).stat().st_size
        files.append({"path": filename, "size_bytes": size})
        weight_map.update(dict.fromkeys(names, filename))
        del tensors
    if sharded:
        (output_dir / "model.safetensors.index.json").write_text(
            json.dumps(
                {"metadata": {"total_size": total_size}, "weight_map": weight_map},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return files, total_size


def _write_tokenizer(output_dir: Path, vocab_file: Path, vocab_size: int) -> None:
    tokenizer = build_fast_tokenizer(vocab_file, vocab_size)
    tokenizer.save(str(output_dir / "tokenizer.json"), pretty=True)
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "added_tokens_decoder": {
                    "0": {
                        "content": END_TOKEN,
                        "lstrip": False,
                        "normalized": False,
                        "rstrip": False,
                        "single_word": False,
                        "special": True,
                    }
                },
                "backend": "tokenizers",
                "bos_token": END_TOKEN,
                "eos_token": END_TOKEN,
                "pad_token": END_TOKEN,
                "padding_side": "left",
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": END_TOKEN,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _validated_checkpoint_path(checkpoint: Path, source_filename: str | None) -> Path:
    checkpoint = checkpoint.expanduser()
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    logical_filename = source_filename or checkpoint.name
    if Path(
        logical_filename
    ).name != logical_filename or not logical_filename.lower().endswith(".pth"):
        raise ValueError("checkpoint must be an original .pth file")
    return checkpoint.resolve()


def inspect_checkpoint(
    checkpoint: Path,
    *,
    dtype: str = "preserve",
    source_filename: str | None = None,
    max_shard_size: str | int = "5GB",
) -> ConversionResult:
    import torch  # pyright: ignore[reportMissingImports]

    checkpoint = _validated_checkpoint_path(checkpoint, source_filename)
    native = _load_checkpoint(checkpoint)
    source_dtypes = _source_float_dtypes(native)
    target_dtype, explicit_cast = _target_dtype(source_dtypes, dtype)
    source_parameter_count = sum(tensor.numel() for tensor in native.values())
    converted = _convert_native(native)
    before = set(converted)
    config = _infer_config(converted, target_dtype)
    synthesized = tuple(sorted(set(converted) - before))
    target_element_size = torch.empty(
        (), dtype=getattr(torch, target_dtype)
    ).element_size()
    maximum = _size_bytes(max_shard_size)
    oversized = {
        name: tensor.numel()
        * (target_element_size if tensor.is_floating_point() else tensor.element_size())
        for name, tensor in converted.items()
        if tensor.numel()
        * (target_element_size if tensor.is_floating_point() else tensor.element_size())
        > maximum
    }
    if oversized:
        name, size = next(iter(oversized.items()))
        raise ValueError(
            f"tensor {name!r} requires {size} bytes, larger than shard limit {maximum}"
        )
    weight_bytes = sum(
        tensor.numel()
        * (target_element_size if tensor.is_floating_point() else tensor.element_size())
        for tensor in converted.values()
    )
    return ConversionResult(
        config=config,
        source_float_dtypes=source_dtypes,
        target_float_dtype=target_dtype,
        explicit_cast=explicit_cast,
        source_parameter_count=source_parameter_count,
        serialized_parameter_count=sum(tensor.numel() for tensor in converted.values()),
        tensor_count=len(converted),
        synthesized_tensors=synthesized,
        weight_bytes=weight_bytes,
        weight_files=(),
    )


def convert_into(
    checkpoint: Path,
    output_dir: Path,
    *,
    vocab_file: Path | None = None,
    dtype: str = "preserve",
    max_shard_size: str | int = "5GB",
    source_filename: str | None = None,
) -> ConversionResult:
    checkpoint = _validated_checkpoint_path(checkpoint, source_filename)
    vocab_file = (
        vocab_file or asset_path("vocab/rwkv_vocab_v20230424.txt")
    ).expanduser()
    if vocab_file.is_symlink() or not vocab_file.is_file():
        raise FileNotFoundError(f"vocabulary does not exist: {vocab_file}")
    vocab_file = vocab_file.resolve()
    _size_bytes(max_shard_size)
    output_dir = output_dir.expanduser().resolve()
    if not output_dir.is_dir() or any(output_dir.iterdir()):
        raise ValueError("conversion destination must be an existing empty directory")

    native = _load_checkpoint(checkpoint)
    source_dtypes = _source_float_dtypes(native)
    target_dtype, explicit_cast = _target_dtype(source_dtypes, dtype)
    source_parameter_count = sum(tensor.numel() for tensor in native.values())
    converted = _convert_native(native)
    before = set(converted)
    config = _infer_config(converted, target_dtype)
    synthesized = tuple(sorted(set(converted) - before))
    weight_files, weight_bytes = _save_sharded_safetensors(
        converted, output_dir, max_shard_size, target_dtype
    )
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "generation_config.json").write_text(
        json.dumps(
            {
                "bos_token_id": 0,
                "eos_token_id": 0,
                "pad_token_id": 0,
                "use_cache": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_tokenizer(output_dir, vocab_file, int(config["vocab_size"]))
    serialized_parameter_count = sum(tensor.numel() for tensor in converted.values())
    return ConversionResult(
        config=config,
        source_float_dtypes=source_dtypes,
        target_float_dtype=target_dtype,
        explicit_cast=explicit_cast,
        source_parameter_count=source_parameter_count,
        serialized_parameter_count=serialized_parameter_count,
        tensor_count=len(converted),
        synthesized_tensors=synthesized,
        weight_bytes=weight_bytes,
        weight_files=tuple(weight_files),
    )


def convert_checkpoint(
    checkpoint: Path,
    output_dir: Path,
    *,
    vocab_file: Path | None = None,
    dtype: str = "preserve",
    max_shard_size: str | int = "5GB",
) -> ConversionResult:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite conversion output: {output_dir}")
    output_dir.mkdir(parents=True)
    try:
        return convert_into(
            checkpoint,
            output_dir,
            vocab_file=vocab_file,
            dtype=dtype,
            max_shard_size=max_shard_size,
            source_filename=checkpoint.name,
        )
    except Exception:
        import shutil

        shutil.rmtree(output_dir)
        raise


__all__ = [
    "ConversionResult",
    "convert_checkpoint",
    "convert_into",
    "inspect_checkpoint",
]
