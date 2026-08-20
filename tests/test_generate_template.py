from __future__ import annotations

import importlib.util
import sys
import types
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any

TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "src/rwkv_publisher/templates/inference_generate.py.template"
)


class FakeTensor:
    def __init__(self, values: list[Any]) -> None:
        self.values = values

    @property
    def ndim(self) -> int:
        return 2 if self.values and isinstance(self.values[0], list) else 1

    @property
    def shape(self) -> tuple[int, ...]:
        if self.ndim == 2:
            return (len(self.values), len(self.values[0]))
        return (len(self.values),)

    def unsqueeze(self, dimension: int):
        assert dimension == 0
        return FakeTensor([self.values])

    def to(self, device: str):
        del device
        return self

    def tolist(self) -> list[Any]:
        return self.values

    def __getitem__(self, key: Any):
        if isinstance(key, tuple):
            row, columns = key
            return FakeTensor(self.values[row][columns])
        return self.values[key]


class FakeTokenizer:
    def __init__(self) -> None:
        self.chat_calls: list[dict[str, Any]] = []

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any):
        self.chat_calls.append({"messages": messages, **kwargs})
        return types.SimpleNamespace(input_ids=FakeTensor([[11, 12]]))

    def decode(self, token_ids: Any, **kwargs: Any) -> str:
        del kwargs
        values = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
        if values == [21, 22, 0]:
            return ">\nplan</think>\nanswer\n\nUser:"
        return ">\npartial\n\nUser:" if 22 in values else ">\npartial"


class FakeModel:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def generate(self, **kwargs: Any):
        self.kwargs = kwargs
        return FakeTensor([[11, 12, 21, 22, 0]])


def _fake_torch() -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.__dict__.update(
        {
            "bfloat16": "bfloat16",
            "float16": "float16",
            "float32": "float32",
            "LongTensor": FakeTensor,
            "FloatTensor": FakeTensor,
            "ones_like": lambda value: value,
            "manual_seed": lambda seed: None,
            "inference_mode": lambda: lambda function: function,
        }
    )
    return torch


def _fake_transformers() -> types.ModuleType:
    transformers = types.ModuleType("transformers")

    class StoppingCriteria:
        pass

    class StoppingCriteriaList(list):
        pass

    transformers.__dict__.update(
        {
            "StoppingCriteria": StoppingCriteria,
            "StoppingCriteriaList": StoppingCriteriaList,
        }
    )
    return transformers


def _load_template_module(monkeypatch: Any):
    loader = types.ModuleType("model_loader")
    loader.__dict__["load_model_and_tokenizer"] = lambda *args, **kwargs: (
        FakeModel(),
        FakeTokenizer(),
    )
    monkeypatch.setitem(sys.modules, "_rwkv7_release_inference.model_loader", loader)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers())
    source_loader = SourceFileLoader("generated_inference", str(TEMPLATE))
    spec = importlib.util.spec_from_loader("generated_inference", source_loader)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generate_template_uses_chat_and_rwkv_stops(monkeypatch: Any) -> None:
    module = _load_template_module(monkeypatch)
    tokenizer = FakeTokenizer()
    model = FakeModel()

    completion = module.generate_completion(
        model,
        tokenizer,
        [{"role": "user", "content": "hello"}],
        device="cpu",
        max_new_tokens=32,
        temperature=1.0,
        top_p=0.5,
        thinking=True,
    )

    assert completion == "<think>\nplan</think>\nanswer"
    assert tokenizer.chat_calls[0]["add_generation_prompt"] is True
    assert tokenizer.chat_calls[0]["thinking"] is True
    assert model.kwargs["eos_token_id"] == 0
    assert model.kwargs["pad_token_id"] == 0
    assert model.kwargs["temperature"] == 1.0
    assert model.kwargs["top_p"] == 0.5


def test_completion_reconstructs_thinking_prefix(monkeypatch: Any) -> None:
    module = _load_template_module(monkeypatch)

    assert module._assistant_content(">\nanswer", thinking=False) == "answer"
    assert (
        module._assistant_content(">\nprivate reasoning</think>\nanswer", thinking=True)
        == "<think>\nprivate reasoning</think>\nanswer"
    )
    assert (
        module._assistant_content(">\nunfinished reasoning", thinking=True)
        == "<think>\nunfinished reasoning"
    )
    assert (
        module._assistant_content(
            ">\nunfinished reasoning", thinking=True, close_incomplete=True
        )
        == "<think>\nunfinished reasoning\n</think>"
    )
    assert (
        module._assistant_content(">\nExplain <think> tags", thinking=False)
        == "Explain <think> tags"
    )
    assert (
        module._assistant_content(">\n<think> tags are literal", thinking=False)
        == "<think> tags are literal"
    )


def test_generation_closes_thinking_only_at_token_limit(monkeypatch: Any) -> None:
    module = _load_template_module(monkeypatch)

    class BudgetModel:
        def generate(self, **kwargs: Any):
            count = kwargs["max_new_tokens"]
            return FakeTensor([[11, 12, *([21] * count)]])

    completion = module.generate_completion(
        BudgetModel(),
        FakeTokenizer(),
        [{"role": "user", "content": "hello"}],
        device="cpu",
        max_new_tokens=3,
        temperature=0,
        top_p=0.5,
        thinking=True,
    )

    assert completion == "<think>\npartial\n</think>"


def test_stop_criterion_detects_split_stop_text(monkeypatch: Any) -> None:
    module = _load_template_module(monkeypatch)
    criterion = module.StopOnText(FakeTokenizer(), 2, "\n\nUser:")

    assert criterion(FakeTensor([[11, 12, 21]]), None) is False
    assert criterion(FakeTensor([[11, 12, 21, 22]]), None) is True


def test_prompt_file_processes_independent_paragraphs(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    module = _load_template_module(monkeypatch)
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("First prompt\n\nSecond\nline\n", encoding="utf-8")
    seen: list[list[dict[str, str]]] = []

    def fake_generate(
        model: Any,
        tokenizer: Any,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        del model, tokenizer, kwargs
        seen.append(messages)
        return "completion"

    module.__dict__["generate_completion"] = fake_generate
    args = types.SimpleNamespace(
        input_file=str(prompts),
        device="cuda",
        max_new_tokens=8,
        temperature=1.0,
        top_p=0.5,
        thinking=False,
    )
    module._file_prompts(FakeModel(), FakeTokenizer(), args)

    assert seen == [
        [{"role": "user", "content": "First prompt"}],
        [{"role": "user", "content": "Second\nline"}],
    ]
    assert capsys.readouterr().out.count("Completion: completion") == 2


def test_interactive_clear_resets_history(monkeypatch: Any) -> None:
    module = _load_template_module(monkeypatch)
    prompts = iter(["hello", "/clear", "again", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(prompts))
    seen: list[list[dict[str, str]]] = []

    def fake_generate(
        model: Any,
        tokenizer: Any,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        del model, tokenizer, kwargs
        seen.append([dict(message) for message in messages])
        return "answer"

    module.__dict__["generate_completion"] = fake_generate
    args = types.SimpleNamespace(
        device="cuda",
        max_new_tokens=8,
        temperature=1.0,
        top_p=0.5,
        thinking=False,
    )
    module._interactive(FakeModel(), FakeTokenizer(), args)

    assert seen == [
        [{"role": "user", "content": "hello"}],
        [{"role": "user", "content": "again"}],
    ]


def test_interactive_replays_clean_assistant_content(monkeypatch: Any) -> None:
    module = _load_template_module(monkeypatch)
    prompts = iter(["first", "second", "/exit"])
    raw_completions = iter([">\nfirst answer", ">\nsecond answer"])
    monkeypatch.setattr("builtins.input", lambda _: next(prompts))
    seen: list[list[dict[str, str]]] = []

    def fake_generate(
        model: Any,
        tokenizer: Any,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        del model, tokenizer
        seen.append([dict(message) for message in messages])
        return module._assistant_content(
            next(raw_completions), thinking=kwargs["thinking"]
        )

    module.__dict__["generate_completion"] = fake_generate
    args = types.SimpleNamespace(
        device="cuda",
        max_new_tokens=8,
        temperature=1.0,
        top_p=0.5,
        thinking=False,
    )
    module._interactive(FakeModel(), FakeTokenizer(), args)

    assert seen == [
        [{"role": "user", "content": "first"}],
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second"},
        ],
    ]


def test_generate_template_has_no_distributed_scaffolding() -> None:
    source = TEMPLATE.read_text(encoding="utf-8")
    assert "torch.distributed" not in source
    assert "init_process_group" not in source
    assert "model{rank}-mp{world_size}" not in source
    compile(source, str(TEMPLATE), "exec")
