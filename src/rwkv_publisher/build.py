from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from string import Template
from typing import Any

from ._version import VERSION
from .assets import (
    ASSET_SET,
    asset_path,
    asset_root,
    canonical_license_path,
    sha256_file,
    verify_assets,
)
from .conversion import ConversionResult, convert_into, inspect_checkpoint
from .manifest import MANIFEST_SCHEMA, inspect_weights, validate_release, write_manifest
from .profiles import FamilyProfile, load_profiles
from .runtime_export import write_flat_runtime
from .source import ResolvedSource, resolve_source

KNOWN_PARAMETER_RANGES = {
    "0.1": (100_000_000, 200_000_000),
    "0.4": (300_000_000, 500_000_000),
    "1.5": (1_300_000_000, 1_700_000_000),
    "2.9": (2_500_000_000, 3_300_000_000),
    "7.2": (6_500_000_000, 8_000_000_000),
    "13.3": (12_000_000_000, 15_000_000_000),
}
OFFICIAL_ORGANIZATION = "BlinkDL"


@dataclass(frozen=True)
class BuildPlan:
    source: ResolvedSource
    model_name: str | None
    output: Path
    dtype: str
    max_shard_size: str
    asset_set: str


def _template(name: str) -> Template:
    path = Path(__file__).resolve().parent / "templates" / name
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"publisher template does not exist: {name}")
    return Template(path.read_text(encoding="utf-8"))


def _parameter_label(parameter_count: int) -> str:
    matches = [
        label
        for label, (lower, upper) in KNOWN_PARAMETER_RANGES.items()
        if lower <= parameter_count <= upper
    ]
    if len(matches) != 1:
        raise ValueError(
            f"could not infer one public parameter label for {parameter_count} parameters"
        )
    return matches[0]


def _release_date(filename: str) -> str:
    values = sorted(set(re.findall(r"(?<!\d)((?:19|20)\d{6})(?!\d)", filename)))
    if len(values) != 1:
        raise ValueError("checkpoint filename must contain exactly one YYYYMMDD date")
    value = values[0]
    date.fromisoformat(f"{value[:4]}-{value[4:6]}-{value[6:]}")
    return value


def _identity(
    source: ResolvedSource, serialized_parameters: int | None = None
) -> tuple[str, str, int | None]:
    if source.profile is not None:
        return (
            source.profile.parameter_label,
            source.profile.release_date,
            source.profile.context_length,
        )
    if serialized_parameters is None:
        raise ValueError("unknown checkpoint identity requires weight inspection")
    return _parameter_label(serialized_parameters), _release_date(source.filename), None


def _yaml(values: tuple[str, ...]) -> str:
    return "  []" if not values else "\n".join(f'  - "{value}"' for value in values)


def _source_url(source: ResolvedSource) -> str:
    if source.reference and source.revision:
        owner, repo, path = source.reference.split("/", 2)
        return f"https://huggingface.co/{owner}/{repo}/blob/{source.revision}/{path}"
    return ""


def _loading_example(repo_id: str, dtype: str) -> str:
    return (
        "```python\n"
        "import torch\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n\n"
        f'model_id = "{repo_id}"\n'
        "tokenizer = AutoTokenizer.from_pretrained(model_id)\n"
        f"model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.{dtype})\n"
        "```"
    )


def _adapt_chat_template(text: str) -> str:
    return text.replace(
        "bos_token | default('')", "bos_token | default('', true)", 1
    ).replace("tools | default([])", "tools | default([], true)", 1)


def _profile_claims(
    source: ResolvedSource,
) -> tuple[FamilyProfile | None, tuple[str, ...], tuple[str, ...]]:
    if source.profile is None:
        return None, (), ()
    family = load_profiles().families[source.profile.family]
    return family, family.languages, family.datasets


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _render_card(
    *,
    source: ResolvedSource,
    conversion: ConversionResult,
    parameter_label: str,
    release_date: str,
    context_length: int | None,
) -> str:
    config = conversion.config
    family, languages, datasets = _profile_claims(source)
    license_id = source.profile.license if source.profile else "other"
    model_name = f"RWKV7-{parameter_label}B-{release_date}"
    repo_id = f"{OFFICIAL_ORGANIZATION}/{model_name}"
    return _template("model_card.md.template").substitute(
        license_id=license_id,
        license_badge=(
            '<a href="LICENSE"><img alt="License" '
            'src="https://img.shields.io/badge/License-Apache--2.0-4c8bf5" /></a>'
            if source.profile
            else '<img alt="Weight license not asserted" '
            'src="https://img.shields.io/badge/Weight%20license-not%20asserted-lightgrey" />'
        ),
        languages=_yaml(languages),
        datasets=_yaml(datasets),
        model_name=model_name,
        repo_id=repo_id,
        architecture="Rwkv7ForCausalLM",
        parameters_billion=parameter_label,
        source_parameters=f"{conversion.source_parameter_count:,}",
        serialized_parameters=f"{conversion.serialized_parameter_count:,}",
        synthesized_tensors=str(len(conversion.synthesized_tensors)),
        num_hidden_layers=config["num_hidden_layers"],
        hidden_size=config["hidden_size"],
        intermediate_size=config["intermediate_size"],
        num_heads=config["num_heads"],
        head_size=config["head_dim"],
        vocab_size=config["vocab_size"],
        training_context=(
            f"{context_length} tokens" if context_length else "not declared"
        ),
        dtype=conversion.target_float_dtype,
        cast_note=(
            f"explicit cast from {', '.join(conversion.source_float_dtypes)}"
            if conversion.explicit_cast
            else "source dtype preserved"
        ),
        source_checkpoint_entry=(
            f"[`{source.reference}`]({_source_url(source)})"
            if source.reference
            else f"`{source.filename}` (unregistered local source)"
        ),
        source_sha256=source.sha256,
        loading_example=_loading_example(repo_id, conversion.target_float_dtype),
        training_description=(
            "This checkpoint is a **base model** pretrained with web, code, "
            "synthetic, instruction, chat, and reasoning data. It is suitable "
            "for evaluation, post-training, and fine-tuning; the included chat "
            "template is a prompt interface, not a claim that the checkpoint is "
            "a safety-aligned assistant."
            if family is not None
            else "This is an unregistered checkpoint. The publisher does not "
            "assert its training corpus, language coverage, or post-training "
            "status; consult the source owner before use."
        ),
        license_statement=(
            "The model weights and exported inference bundle are released under "
            f"[{license_id}](LICENSE)."
            if source.profile
            else "The exported inference bundle is licensed under [Apache-2.0](LICENSE). "
            "This publisher does not assert a license for the unregistered model weights."
        ),
    )


def _write_card(
    root: Path,
    *,
    source: ResolvedSource,
    conversion: ConversionResult,
    parameter_label: str,
    release_date: str,
    context_length: int | None,
) -> None:
    card = _render_card(
        source=source,
        conversion=conversion,
        parameter_label=parameter_label,
        release_date=release_date,
        context_length=context_length,
    )
    (root / "README.md").write_text(card, encoding="utf-8")


def _write_inference(
    root: Path,
) -> dict[str, Any]:
    inference = root / "inference"
    runtime_source = asset_root() / "runtime" / ASSET_SET
    runtime = write_flat_runtime(runtime_source, inference)
    for filename, template_name in (
        ("generate.py", "inference_generate.py.template"),
        ("model_loader.py", "inference_model_loader.py.template"),
        ("requirements.txt", "inference_requirements.txt.template"),
    ):
        (inference / filename).write_text(
            _template(template_name).template, encoding="utf-8"
        )
    return runtime.provenance


def plan_build(
    source: str | Path,
    *,
    output: Path = Path("dist"),
    dtype: str = "preserve",
    max_shard_size: str = "5GB",
    source_ref: str | None = None,
    offline: bool = False,
) -> BuildPlan:
    assets = verify_assets()
    resolved = resolve_source(source, source_ref=source_ref, offline=offline)
    model_name = resolved.profile.model_name if resolved.profile else None
    return BuildPlan(
        source=resolved,
        model_name=model_name,
        output=output.expanduser().resolve(),
        dtype=dtype,
        max_shard_size=max_shard_size,
        asset_set=assets["asset_set"],
    )


def build_release(
    source: str | Path,
    *,
    output: Path = Path("dist"),
    dtype: str = "preserve",
    max_shard_size: str = "5GB",
    source_ref: str | None = None,
    offline: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    plan = plan_build(
        source,
        output=output,
        dtype=dtype,
        max_shard_size=max_shard_size,
        source_ref=source_ref,
        offline=offline,
    )
    if dry_run:
        inspection = inspect_checkpoint(
            plan.source.local_path,
            dtype=dtype,
            source_filename=plan.source.filename,
            max_shard_size=max_shard_size,
        )
        parameter_label, release_date, _ = _identity(
            plan.source, inspection.serialized_parameter_count
        )
        model_name = f"RWKV7-{parameter_label}B-{release_date}"
        if plan.source.profile and model_name != plan.source.profile.model_name:
            raise ValueError("checkpoint does not match registered release identity")
        return {
            "dry_run": True,
            "source": plan.source.reference or str(plan.source.local_path),
            "source_sha256": plan.source.sha256,
            "model_name": model_name,
            "output": str(plan.output),
            "dtype": inspection.target_float_dtype,
            "asset_set": plan.asset_set,
            "parameter_count": inspection.serialized_parameter_count,
            "tensor_count": inspection.tensor_count,
            "weight_bytes": inspection.weight_bytes,
        }
    plan.output.mkdir(parents=True, exist_ok=True)
    if plan.model_name and (plan.output / plan.model_name).exists():
        raise FileExistsError(
            f"refusing to overwrite existing release: {plan.output / plan.model_name}"
        )
    asset_lock = verify_assets()
    with tempfile.TemporaryDirectory(
        prefix=".rwkv-publisher-", dir=plan.output
    ) as temporary:
        temporary_root = Path(temporary)
        stage = temporary_root / "release"
        stage.mkdir()
        incomplete = temporary_root / ".incomplete"
        incomplete.write_text("release assembly in progress\n", encoding="utf-8")
        conversion = convert_into(
            plan.source.local_path,
            stage,
            dtype=dtype,
            max_shard_size=max_shard_size,
            source_filename=plan.source.filename,
        )
        if (
            plan.source.local_path.stat().st_size != plan.source.size_bytes
            or sha256_file(plan.source.local_path) != plan.source.sha256
        ):
            raise RuntimeError("source checkpoint changed while the release was built")
        parameter_label, release_date, context_length = _identity(
            plan.source, conversion.serialized_parameter_count
        )
        model_name = f"RWKV7-{parameter_label}B-{release_date}"
        destination = plan.output / model_name
        if destination.exists():
            raise FileExistsError(
                f"refusing to overwrite existing release: {destination}"
            )
        if plan.source.profile and model_name != plan.source.profile.model_name:
            raise ValueError(
                "converted weights do not match registered release identity"
            )
        stage.joinpath("chat_template.jinja").write_text(
            _adapt_chat_template(
                asset_path("templates/chat_template.jinja").read_text(encoding="utf-8")
            ),
            encoding="utf-8",
        )
        stage.joinpath(".gitattributes").write_text(
            _template("gitattributes.template").template, encoding="utf-8"
        )
        shutil.copy2(canonical_license_path(), stage / "LICENSE")
        stage.joinpath("NOTICE").write_text(
            f"RWKV-7 model release\nSource: {plan.source.reference or plan.source.filename}\n"
            "Exported inference bundle licensed under Apache-2.0.\n",
            encoding="utf-8",
        )
        _write_card(
            stage,
            source=plan.source,
            conversion=conversion,
            parameter_label=parameter_label,
            release_date=release_date,
            context_length=context_length,
        )
        repo_id = f"{OFFICIAL_ORGANIZATION}/{model_name}"
        runtime = _write_inference(stage)
        incomplete.unlink()
        facts = inspect_weights(stage)
        profile = plan.source.profile
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "builder": {
                "version": VERSION,
                "asset_set": ASSET_SET,
                "assets_sha256": asset_lock["combined_sha256"],
            },
            "source": {
                "kind": plan.source.kind,
                "reference": plan.source.reference,
                "revision": plan.source.revision,
                "filename": plan.source.filename,
                "sha256": plan.source.sha256,
                "size_bytes": plan.source.size_bytes,
            },
            "profile": {
                "checkpoint": profile.id if profile else None,
                "family": profile.family if profile else None,
            },
            "identity": {
                "parameter_label": parameter_label,
                "release_date": release_date,
                "training_context_length": context_length,
            },
            "conversion": {
                "source_float_dtypes": list(conversion.source_float_dtypes),
                "target_float_dtypes": [conversion.target_float_dtype],
                "explicit_cast": conversion.explicit_cast,
                "source_parameter_count": conversion.source_parameter_count,
                "serialized_parameter_count": conversion.serialized_parameter_count,
                "tensor_count": conversion.tensor_count,
                "synthesized_tensors": list(conversion.synthesized_tensors),
                "tensor_map_sha256": facts.tensor_map_sha256,
            },
            "runtime": runtime,
        }
        write_manifest(stage, manifest)
        named_stage = temporary_root / model_name
        stage.rename(named_stage)
        validate_release(named_stage)
        os.rename(named_stage, destination)
        _make_read_only(destination)
    return {
        "dry_run": False,
        "release": str(destination),
        "model_name": model_name,
        "repo_id": repo_id,
        "source_sha256": plan.source.sha256,
        "dtype": conversion.target_float_dtype,
        "parameter_count": conversion.serialized_parameter_count,
        "tensor_count": conversion.tensor_count,
    }


__all__ = ["BuildPlan", "build_release", "plan_build"]
