# Changelog

## 0.2.0

- Replaced the TOML-based `convert` / `init` / `build` workflow with
  `rwkv-publisher build SOURCE`.
- Made `publish RELEASE_DIR` immediate, with `--dry-run` as the explicit
  simulation mode and optional owner override through `--repo`.
- Embedded and cryptographically locked the official profiles, vocabulary, chat
  template, and validated inference runtime.
- Preserved source dtype by default and recorded every explicit cast and
  synthesized tensor in one semantic release manifest.
- Bundled the Transformers 5.15 RWKV-7 configuration and modeling modules with
  strict `AutoConfig`, `AutoModel`, and `AutoModelForCausalLM` remote mappings;
  the Fast tokenizer remains native.
- Removed the spurious chat BOS token and reconstructed the partial RWKV thinking
  prefix before displaying or retaining assistant replies.
- Reduced the optional runtime to five files under `inference/`: one generated
  `runtime.py`, one merged `kernel.py`, the loader, CLI, and requirements. Removed
  generated compatibility packages, internal module files, duplicated docs, and
  duplicated license files.
- Added parent-locked atomic Hub commits and immutable remote verification.
- Added an optional metadata-only TOML, explicit `--profile`, and a terminal
  profile selector used only for unknown checkpoints. Non-interactive builds
  remain prompt-free and default to no unsupported claims.

This is an intentional clean break. Version 0.2 does not read `release.toml` and
does not provide the 0.1 `convert`, `init`, `--config`, or `--execute` interfaces.
