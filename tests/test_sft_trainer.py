from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from datasets import Dataset
from peft import LoraConfig
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from rwkv_publisher.assets import asset_path
from rwkv_publisher.remote_code import REMOTE_AUTO_MAP, build_model_code


def _write_tiny_tokenizer(root: Path) -> None:
    vocabulary = {
        "<pad>": 0,
        "<eos>": 1,
        "<unk>": 2,
        "User": 3,
        "Assistant": 4,
        "System": 5,
        ":": 6,
        "hello": 7,
        "hi": 8,
        "again": 9,
        "answer": 10,
        "briefly": 11,
        "<": 12,
        ">": 13,
        "/": 14,
        "think": 15,
    }
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
    )
    tokenizer.chat_template = asset_path("templates/chat_template.jinja").read_text(
        encoding="utf-8"
    )
    tokenizer.save_pretrained(root)


@pytest.fixture(scope="module")
def tiny_rwkv7_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("tiny-rwkv7")
    for filename, source in build_model_code().items():
        root.joinpath(filename).write_text(source, encoding="utf-8")

    config_dict = {
        "model_type": "rwkv7",
        "architectures": ["Rwkv7ForCausalLM"],
        "auto_map": REMOTE_AUTO_MAP,
        "vocab_size": 16,
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "head_dim": 4,
        "num_heads": 2,
        "intermediate_size": 16,
        "decay_low_rank_dim": 4,
        "a_low_rank_dim": 4,
        "v_low_rank_dim": 4,
        "gate_low_rank_dim": 4,
        "use_cache": True,
        "wkv_implementation": "eager",
        "bos_token_id": 1,
        "eos_token_id": 1,
        "pad_token_id": 0,
    }
    root.joinpath("config.json").write_text(json.dumps(config_dict), encoding="utf-8")
    _write_tiny_tokenizer(root)

    config = AutoConfig.from_pretrained(
        root, trust_remote_code=True, local_files_only=True
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model.save_pretrained(root, safe_serialization=True)
    return root


def _load_model(root: Path) -> Any:
    return AutoModelForCausalLM.from_pretrained(
        root, trust_remote_code=True, local_files_only=True
    )


def _load_tokenizer(root: Path) -> Any:
    return AutoTokenizer.from_pretrained(root, local_files_only=True)


def _training_args(output_dir: Path, **kwargs) -> SFTConfig:
    gradient_checkpointing = kwargs.pop("gradient_checkpointing", False)
    return SFTConfig(
        output_dir=str(output_dir),
        max_steps=1,
        per_device_train_batch_size=2,
        learning_rate=1e-3,
        max_length=32,
        use_cpu=True,
        gradient_checkpointing=gradient_checkpointing,
        dataloader_pin_memory=False,
        disable_tqdm=True,
        logging_strategy="no",
        save_strategy="no",
        report_to="none",
        **kwargs,
    )


def test_sft_contract_and_default_chunked_nll(
    tiny_rwkv7_repo: Path, tmp_path: Path
) -> None:
    model = _load_model(tiny_rwkv7_repo)
    input_ids = torch.tensor([[3, 6, 7, 4, 6]], dtype=torch.long)

    with torch.no_grad():
        base_output = model.rwkv7(input_ids=input_ids, use_cache=False)
        labelled_output = model(input_ids=input_ids, labels=input_ids)
        packed_from_positions = model.rwkv7(
            input_ids=input_ids,
            position_ids=torch.tensor([[0, 1, 2, 0, 1]]),
            use_cache=False,
        )
        packed_from_boundaries = model.rwkv7(
            input_ids=input_ids,
            cu_seq_lens=torch.tensor([0, 3, 5]),
            use_cache=False,
        )

    assert base_output.past_key_values is None
    assert labelled_output.state is None
    dead_parameter_names = {
        "rwkv7.blocks.0.att.v0",
        "rwkv7.blocks.0.att.v1",
        "rwkv7.blocks.0.att.v2",
    }
    assert dead_parameter_names.isdisjoint(dict(model.named_parameters()))
    assert dead_parameter_names <= set(dict(model.named_buffers()))
    assert dead_parameter_names <= set(model.state_dict())
    torch.testing.assert_close(
        packed_from_positions.last_hidden_state,
        packed_from_boundaries.last_hidden_state,
    )

    args = _training_args(tmp_path / "default", gradient_checkpointing=True)
    assert args.loss_type == "chunked_nll"
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_dict(
            {
                "text": [
                    "User: hello\n\nAssistant: hi",
                    "User: again\n\nAssistant: answer",
                ]
            }
        ),
        processing_class=_load_tokenizer(tiny_rwkv7_repo),
    )
    assert trainer.train().global_step == 1


def test_sft_packing_with_assistant_only_loss(
    tiny_rwkv7_repo: Path, tmp_path: Path
) -> None:
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ],
            },
            {
                "messages": [
                    {"role": "user", "content": "again"},
                    {"role": "assistant", "content": "answer briefly"},
                ],
            },
        ]
    )
    trainer = SFTTrainer(
        model=_load_model(tiny_rwkv7_repo),
        args=_training_args(
            tmp_path / "packed", packing=True, assistant_only_loss=True
        ),
        train_dataset=dataset,
        processing_class=_load_tokenizer(tiny_rwkv7_repo),
    )
    batch = next(iter(trainer.get_train_dataloader()))
    assert "position_ids" in batch
    assert (batch["position_ids"] == 0).sum() >= 2
    assert (batch["labels"] == -100).any()
    assert trainer.train().global_step == 1


def test_sft_peft_lora_with_explicit_rwkv_targets(
    tiny_rwkv7_repo: Path, tmp_path: Path
) -> None:
    trainer = SFTTrainer(
        model=_load_model(tiny_rwkv7_repo),
        args=_training_args(tmp_path / "peft"),
        train_dataset=Dataset.from_dict(
            {
                "text": [
                    "User: hello\n\nAssistant: hi",
                    "User: again\n\nAssistant: answer",
                ]
            }
        ),
        processing_class=_load_tokenizer(tiny_rwkv7_repo),
        peft_config=LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["receptance", "key", "value", "output"],
        ),
    )
    trained_model = trainer.model
    assert trained_model is not None
    trainable_names = [
        name
        for name, parameter in trained_model.named_parameters()
        if parameter.requires_grad
    ]
    assert trainable_names
    assert all("lora_" in name for name in trainable_names)
    assert trainer.train().global_step == 1
