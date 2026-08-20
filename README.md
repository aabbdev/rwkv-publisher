# rwkv-publisher

`rwkv-publisher` turns an original RWKV-7 `.pth` checkpoint into a complete,
auditable Hugging Face model repository, then publishes it as one verified commit.

The workflow has two commands:

```bash
rwkv-publisher build /models/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth
rwkv-publisher publish dist/RWKV7-0.1B-20241210 \
  --repo OWNER/RWKV7-0.1B-20241210
```

No project file, manual conversion, vocabulary path, runtime path, hash, parameter
label, context length, or license flag is required.

## Install

Python 3.12 is required.

```bash
pip install rwkv-publisher
hf auth login  # only required for publishing
```

From this checkout:

```bash
uv sync
uv run rwkv-publisher --help
```

## Build

```bash
rwkv-publisher build SOURCE
```

`SOURCE` can be:

- a local `.pth` checkpoint;
- a registered Hugging Face checkpoint path;
- a supported Hugging Face repository containing one canonical checkpoint;
- an HTTPS Hugging Face checkpoint URL.

The build automatically:

1. resolves and verifies the source;
2. infers and validates the RWKV-7 architecture;
3. preserves the source floating dtype by default;
4. writes bounded safetensors shards directly into temporary final staging;
5. generates the config, bundled Transformers 5.15 RWKV-7 code, tokenizer, chat
   template, model card, and optional optimized runtime;
6. reconstructs and validates every generated artifact;
7. atomically creates `dist/RWKV7-<size>B-<YYYYMMDD>`.

Existing destinations are never overwritten. Failed builds remove only their own
temporary staging directory.

### Build options

```text
--output DIR              release parent directory (default: dist)
--dtype DTYPE             preserve, float32, float16, or bfloat16
--max-shard-size SIZE     logical shard limit (default: 5GB)
--source-ref OWNER/REPO/PATH
                          immutable provenance for an unregistered local source
--config FILE             optional metadata-only TOML
--profile NAME            metadata family for an unregistered checkpoint
--no-input                disable the interactive profile selector
--offline                 forbid network access
--dry-run                 inspect and validate without serializing weights
--json                    machine-readable output
```

An explicit dtype different from the checkpoint is recorded as a numerical cast.
Mixed or unsupported source dtypes require an explicit choice.

### Optional metadata profiles

Known checkpoints select their locked metadata profile automatically. An unknown
checkpoint opened from an interactive terminal displays a small profile selector;
press Enter to publish without language, dataset, training-context, or weight-license
claims.

Automation never receives a prompt. Use `--no-input`, `--json`, `--profile`, or an
optional TOML file:

```toml
schema = 1
profile = "world-v2.8"

[metadata]
languages = ["en", "fr"]
datasets = ["organization/dataset"]
context_length = 4096
license = "apache-2.0"
```

```bash
rwkv-publisher build model-20260806.pth \
  --config examples/rwkv-publisher.toml
```

The TOML intentionally cannot set paths, hashes, dtype, or parameter size.
`context_length` is documented as training context, never as a recurrent inference
limit. `license` is a weight-license identifier supplied as metadata; the generated
runtime remains Apache-2.0. `--profile` overrides the TOML profile. Values that
conflict with a recognized checkpoint are rejected. A file containing only
`schema = 1` explicitly selects no claims for an unknown checkpoint and suppresses
the terminal menu.

## Publish

Publishing is immediate:

```bash
rwkv-publisher publish dist/RWKV7-1.5B-20260710 \
  --repo OWNER/RWKV7-1.5B-20260710
```

Use `--dry-run` to perform local validation without contacting the Hub:

```bash
rwkv-publisher publish dist/RWKV7-1.5B-20260710 \
  --repo OWNER/RWKV7-1.5B-20260710 \
  --private \
  --dry-run
```

The destination basename must equal the release basename; only the owner may
change. Publication:

- validates local semantics and hashes again;
- observes and locks the remote parent commit;
- adds current files and deletes stale paths in one atomic commit;
- captures the immutable commit SHA;
- verifies the exact remote tree, LFS objects, sizes, and file hashes.

Credentials come from `hf auth login`; tokens are never command-line arguments.

## Generated repository

The root is a standalone Transformers model repository. It embeds the audited
RWKV-7 configuration and modeling modules and maps the three model auto-classes to
them. Model loading therefore uses `trust_remote_code=True`; the Fast tokenizer
remains native and has no remote-code mapping.

```text
RWKV7-<size>B-<YYYYMMDD>/
├── README.md
├── LICENSE
├── NOTICE
├── config.json
├── configuration_rwkv7.py
├── generation_config.json
├── modeling_rwkv7.py
├── chat_template.jinja
├── model*.safetensors
├── model.safetensors.index.json  # only when sharded
├── tokenizer.json
├── tokenizer_config.json
├── release-manifest.json
└── inference/
    ├── generate.py
    ├── model_loader.py
    ├── runtime.py
    ├── kernel.py
    └── requirements.txt
```

The five-file `inference/` directory is optional and self-contained:

- `generate.py` is the chat and prompt CLI;
- `model_loader.py` streams native shards into the optimized model;
- `runtime.py` contains the isolated generated Python runtime;
- `kernel.py` contains the isolated state and decode TileLang namespaces;
- `requirements.txt` pins the validated runtime dependencies.

The repository-level `LICENSE` and `NOTICE` apply to `inference/`; they are not
duplicated inside the directory.

Run the optimized interface with:

```bash
python inference/generate.py --model . --backend auto --interactive
```

`auto` uses validated exact optimized boundaries and falls back to pure PyTorch
when TileLang, CUDA, dtype, architecture, or shape support is unavailable.

Use Transformers 5.15 or newer and pass `trust_remote_code=True` to `AutoConfig`,
`AutoModel`, or `AutoModelForCausalLM`. Review the two root Python files and pin a
Hub revision in production. `AutoTokenizer.from_pretrained(...)` does not require
remote-code trust.

## Integrity model

Publisher assets and official checkpoint profiles are embedded and SHA-256
locked. The schema-6 manifest is destination-neutral and records source identity,
dtype behavior, parameter counts, synthesized compatibility tensors, model-code
and runtime provenance, and every released file.

Validation does not trust editable manifest declarations: it rebuilds the
tokenizer, bundled model code, and optimized runtime from locked assets, rerenders
templates, rechecks safetensors headers and shard paths, and compares the canonical
license and model card.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
uv build
```

Locked assets are updated only through `scripts/update_assets.py`.

## License

The publisher and generated optimized inference runtime are licensed under
Apache-2.0. A model release reports the weight license proven by its locked source
profile; unregistered checkpoints do not receive an invented weight-license claim.
