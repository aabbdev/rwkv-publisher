from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers

END_TOKEN = "<|endoftext|>"
INVALID_TOKEN_PREFIX = "\ue000"
INVALID_TOKEN_SUFFIX = "\ue001"


def read_rwkv_vocab(vocab_file: Path) -> dict[int, bytes]:
    tokens: dict[int, bytes] = {}
    token_ids: dict[bytes, int] = {}
    for row in vocab_file.read_text(encoding="utf-8").splitlines():
        if not row:
            continue
        try:
            first_space = row.index(" ")
            last_space = row.rindex(" ")
            token_id = int(row[:first_space])
            expected_length = int(row[last_space + 1 :])
            token = ast.literal_eval(row[first_space:last_space].strip())
        except (SyntaxError, ValueError) as error:
            raise ValueError(f"invalid RWKV vocabulary row: {row!r}") from error
        token = token.encode("utf-8") if isinstance(token, str) else token
        if not isinstance(token, bytes) or len(token) != expected_length:
            raise ValueError(f"invalid RWKV vocabulary row: {row!r}")
        if token_id <= 0 or token_id in tokens or token in token_ids:
            raise ValueError(
                f"invalid or duplicate RWKV token: id={token_id}, token={token!r}"
            )
        tokens[token_id] = token
        token_ids[token] = token_id

    if not tokens:
        raise ValueError("RWKV vocabulary is empty")
    missing_bytes = [byte for byte in range(256) if bytes([byte]) not in token_ids]
    if missing_bytes:
        raise ValueError(
            f"RWKV vocabulary is missing single-byte tokens: {missing_bytes}"
        )
    for token_id, token in tokens.items():
        for endpoint in range(1, len(token)):
            prefix_id = token_ids.get(token[:endpoint])
            if prefix_id is not None and prefix_id > token_id:
                raise ValueError(
                    "RWKV token ranks cannot use greedy longest-prefix tokenization: "
                    f"id {prefix_id} precedes id {token_id}"
                )
    return tokens


def _bytes_to_unicode() -> dict[int, str]:
    values = list(range(ord("!"), ord("~") + 1))
    values += list(range(ord("¡"), ord("¬") + 1))
    values += list(range(ord("®"), ord("ÿ") + 1))
    characters = list(values)
    extra = 0
    for byte in range(256):
        if byte not in values:
            values.append(byte)
            characters.append(256 + extra)
            extra += 1
    return dict(zip(values, map(chr, characters), strict=True))


def build_fast_tokenizer(vocab_file: Path, model_vocab_size: int) -> Tokenizer:
    tokens = read_rwkv_vocab(vocab_file)
    inferred_size = max(tokens) + 1
    if model_vocab_size < inferred_size:
        raise ValueError(
            f"model vocab size {model_vocab_size} is smaller than tokenizer size {inferred_size}"
        )
    byte_encoder = _bytes_to_unicode()
    vocab = {END_TOKEN: 0}
    for token_id, token in tokens.items():
        vocab["".join(byte_encoder[byte] for byte in token)] = token_id
    for token_id in range(model_vocab_size):
        if token_id not in tokens and token_id != 0:
            vocab[f"{INVALID_TOKEN_PREFIX}{token_id}{INVALID_TOKEN_SUFFIX}"] = token_id
    if len(vocab) != model_vocab_size:
        raise ValueError(
            "RWKV vocabulary does not map one token to every model token id"
        )

    tokenizer = Tokenizer(
        models.WordPiece(
            vocab=vocab,
            unk_token=END_TOKEN,
            max_input_chars_per_word=2**31 - 1,
            continuing_subword_prefix="",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        trim_offsets=True,
        use_regex=False,
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens([AddedToken(END_TOKEN, special=True)])
    return tokenizer


def encode_reference(text: str, tokens: dict[int, bytes]) -> list[int]:
    token_ids = {token: token_id for token_id, token in tokens.items()}
    candidates: dict[tuple[int, int], list[bytes]] = {}
    for token_id, token in sorted(tokens.items(), reverse=True):
        del token_id
        if len(token) >= 2:
            candidates.setdefault((token[0], token[1]), []).append(token)

    data = text.encode("utf-8")
    encoded: list[int] = []
    position = 0
    while position < len(data):
        token = data[position : position + 1]
        if position + 1 < len(data):
            token = next(
                (
                    candidate
                    for candidate in candidates.get(
                        (data[position], data[position + 1]), []
                    )
                    if data.startswith(candidate, position)
                ),
                token,
            )
        encoded.append(token_ids[token])
        position += len(token)
    return encoded


def vocab_sha256(vocab_file: Path) -> str:
    return hashlib.sha256(vocab_file.read_bytes()).hexdigest()
