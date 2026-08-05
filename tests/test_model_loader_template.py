from __future__ import annotations

import importlib.util
import sys
import types
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "src/rwkv_publisher/templates/inference_model_loader.py.template"
)


def _load_template(monkeypatch: pytest.MonkeyPatch, snapshot: Path):
    torch = types.ModuleType("torch")
    torch.__dict__.update({"dtype": object, "Tensor": object})
    hub = types.ModuleType("huggingface_hub")
    hub.__dict__["snapshot_download"] = lambda model: str(snapshot)
    safetensors = types.ModuleType("safetensors")
    safetensors_torch = types.ModuleType("safetensors.torch")
    safetensors_torch.__dict__["load_file"] = lambda *args, **kwargs: {}
    transformers = types.ModuleType("transformers")
    transformers.__dict__.update({"AutoTokenizer": object, "PreTrainedConfig": object})
    package = types.ModuleType("inference")
    package.__path__ = []  # type: ignore[attr-defined]
    runtime = types.ModuleType("inference.runtime")
    runtime.__dict__.update({"RWKV7Config": object, "RWKV7ForCausalLM": object})
    for name, module in {
        "torch": torch,
        "huggingface_hub": hub,
        "safetensors": safetensors,
        "safetensors.torch": safetensors_torch,
        "transformers": transformers,
        "inference": package,
        "inference.runtime": runtime,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    loader = SourceFileLoader("inference.model_loader", str(TEMPLATE))
    spec = importlib.util.spec_from_loader("inference.model_loader", loader)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hub_snapshot_allows_cache_weight_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blobs = tmp_path / "blobs"
    snapshot = tmp_path / "snapshots/revision"
    blobs.mkdir()
    snapshot.mkdir(parents=True)
    target = blobs / "weight"
    target.write_bytes(b"weights")
    (snapshot / "model.safetensors").symlink_to(target)
    module = _load_template(monkeypatch, snapshot)

    resolved, from_hub = module._resolve_model("BlinkDL/model")
    plan = module._weight_plan(resolved, hub_snapshot=from_hub)

    assert from_hub is True
    assert plan == [(target.resolve(), None)]


def test_local_weight_symlink_remains_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    target = tmp_path / "weight"
    target.write_bytes(b"weights")
    (model / "model.safetensors").symlink_to(target)
    module = _load_template(monkeypatch, model)

    with pytest.raises(RuntimeError, match="local safetensor must not be a symlink"):
        module._weight_plan(model, hub_snapshot=False)


def test_model_loader_template_compiles() -> None:
    compile(TEMPLATE.read_text(encoding="utf-8"), str(TEMPLATE), "exec")
