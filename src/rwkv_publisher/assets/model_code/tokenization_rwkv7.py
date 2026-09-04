# coding=utf-8
"""Exact linear-time RWKV World tokenizer for Transformers AutoTokenizer."""

from __future__ import annotations

import json
import os
import shutil
from typing import Optional

from transformers.tokenization_utils import PreTrainedTokenizer


VOCAB_FILES_NAMES = {"tokenizer_file": "tokenizer.json"}
END_TOKEN = "<|endoftext|>"
INVALID_TOKEN_PREFIX = "\ue000"
INVALID_TOKEN_SUFFIX = "\ue001"


class _CharEncoding:
    """Minimal Encoding surface used by BatchEncoding.char_to_token."""

    def __init__(self, offsets):
        self.offsets = offsets

    def char_to_token(self, char_index, sequence_index=0):
        if sequence_index != 0:
            return None
        for token_index, offset in enumerate(self.offsets):
            if offset is not None and offset[0] <= char_index < offset[1]:
                return token_index
        return None


def _bytes_to_unicode():
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
    return dict(zip(values, map(chr, characters)))


class Rwkv7Tokenizer(PreTrainedTokenizer):
    """RWKV World longest-prefix tokenizer backed only by tokenizer.json."""

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    padding_side = "left"

    def __init__(
        self,
        tokenizer_file,
        eos_token=END_TOKEN,
        pad_token=END_TOKEN,
        unk_token=END_TOKEN,
        add_bos_token=False,
        **kwargs,
    ):
        if not tokenizer_file or not os.path.isfile(tokenizer_file):
            raise ValueError(f"tokenizer.json does not exist: {tokenizer_file}")
        if add_bos_token:
            raise ValueError("RWKV World has no BOS token")
        self.tokenizer_file = tokenizer_file
        self.add_bos_token = False
        value = json.loads(open(tokenizer_file, encoding="utf-8").read())
        model = value.get("model")
        if not isinstance(model, dict) or model.get("type") != "WordPiece":
            raise ValueError("tokenizer.json is not the locked RWKV WordPiece model")
        if (
            model.get("unk_token") != END_TOKEN
            or model.get("continuing_subword_prefix") != ""
            or value.get("normalizer") is not None
        ):
            raise ValueError("tokenizer.json does not use exact RWKV semantics")
        pre_tokenizer = value.get("pre_tokenizer")
        decoder = value.get("decoder")
        if (
            not isinstance(pre_tokenizer, dict)
            or pre_tokenizer.get("type") != "ByteLevel"
            or pre_tokenizer.get("add_prefix_space") is not False
            or pre_tokenizer.get("use_regex") is not False
            or not isinstance(decoder, dict)
            or decoder.get("type") != "ByteLevel"
        ):
            raise ValueError("tokenizer.json does not use exact ByteLevel semantics")
        vocabulary = model.get("vocab")
        if not isinstance(vocabulary, dict) or not vocabulary:
            raise ValueError("tokenizer.json has no vocabulary")
        if any(not isinstance(token, str) or not isinstance(index, int) for token, index in vocabulary.items()):
            raise TypeError("tokenizer.json vocabulary entry is invalid")
        if set(vocabulary.values()) != set(range(len(vocabulary))):
            raise ValueError("tokenizer.json vocabulary ids are not contiguous")
        if vocabulary.get(END_TOKEN) != 0:
            raise ValueError("tokenizer.json must reserve id 0 for end-of-text")

        byte_decoder = {character: byte for byte, character in _bytes_to_unicode().items()}
        self.encoder = dict(vocabulary)
        self.decoder = {index: token for token, index in vocabulary.items()}
        self._token_bytes = {}
        reverse = {}
        for token, index in vocabulary.items():
            if index == 0:
                continue
            placeholder = f"{INVALID_TOKEN_PREFIX}{index}{INVALID_TOKEN_SUFFIX}"
            if token == placeholder:
                continue
            try:
                raw = bytes(byte_decoder[character] for character in token)
            except KeyError as error:
                raise ValueError("tokenizer.json vocabulary is not ByteLevel encoded") from error
            if raw in reverse:
                raise ValueError("tokenizer.json contains duplicate byte tokens")
            reverse[raw] = index
            self._token_bytes[token] = raw
        missing = [byte for byte in range(256) if bytes([byte]) not in reverse]
        if missing:
            raise ValueError(f"tokenizer.json is missing singleton bytes: {missing}")
        for raw, index in reverse.items():
            for endpoint in range(1, len(raw)):
                prefix = reverse.get(raw[:endpoint])
                if prefix is not None and prefix > index:
                    raise ValueError("token ranks do not support longest-prefix encoding")

        self._children = [{}]
        self._terminal = [None]
        self._max_token_bytes = 1
        for raw, index in sorted(reverse.items(), key=lambda item: item[1]):
            node = 0
            for byte in raw:
                child = self._children[node].get(byte)
                if child is None:
                    child = len(self._children)
                    self._children[node][byte] = child
                    self._children.append({})
                    self._terminal.append(None)
                node = child
            self._terminal[node] = index
            self._max_token_bytes = max(self._max_token_bytes, len(raw))

        if "additional_special_tokens" not in kwargs:
            appended = []
            for item in sorted(value.get("added_tokens", []), key=lambda item: item.get("id", -1)):
                if not isinstance(item, dict):
                    raise TypeError("tokenizer.json added token is invalid")
                content = item.get("content")
                index = item.get("id")
                if not isinstance(content, str) or not isinstance(index, int):
                    raise TypeError("tokenizer.json added token is invalid")
                if content != END_TOKEN:
                    if not item.get("special") or index < len(vocabulary):
                        raise ValueError("only append-only special tokens are supported")
                    appended.append(content)
            if appended:
                kwargs["additional_special_tokens"] = appended
        super().__init__(
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            add_bos_token=self.add_bos_token,
            **kwargs,
        )

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        vocabulary = dict(self.encoder)
        vocabulary.update(self.added_tokens_encoder)
        return vocabulary

    def _encode_bytes(self, data):
        output = []
        position = 0
        while position < len(data):
            node = 0
            cursor = position
            best_id = None
            best_end = position
            limit = min(len(data), position + self._max_token_bytes)
            while cursor < limit:
                child = self._children[node].get(data[cursor])
                if child is None:
                    break
                node = child
                cursor += 1
                token_id = self._terminal[node]
                if token_id is not None and (best_id is None or token_id > best_id):
                    best_id = token_id
                    best_end = cursor
            if best_id is None:
                raise RuntimeError("singleton-byte vocabulary invariant was violated")
            output.append(best_id)
            position = best_end
        return output

    def _encode_bytes_with_offsets(self, data, char_offset, byte_to_char):
        output = []
        position = 0
        while position < len(data):
            node = 0
            cursor = position
            best_id = None
            best_end = position
            limit = min(len(data), position + self._max_token_bytes)
            while cursor < limit:
                child = self._children[node].get(data[cursor])
                if child is None:
                    break
                node = child
                cursor += 1
                token_id = self._terminal[node]
                if token_id is not None and (best_id is None or token_id > best_id):
                    best_id = token_id
                    best_end = cursor
            if best_id is None:
                raise RuntimeError("singleton-byte vocabulary invariant was violated")
            start_char = char_offset + byte_to_char[position]
            end_char = char_offset + byte_to_char[best_end - 1] + 1
            output.append((best_id, (start_char, end_char)))
            position = best_end
        return output

    def _encode_with_offsets(self, text):
        special = sorted(self.added_tokens_encoder, key=lambda token: (-len(token), token))
        output = []
        ordinary_start = 0
        position = 0

        def append_ordinary(segment, char_offset):
            if not segment:
                return
            data = segment.encode("utf-8")
            byte_to_char = []
            for char_index, character in enumerate(segment):
                byte_to_char.extend([char_index] * len(character.encode("utf-8")))
            output.extend(
                self._encode_bytes_with_offsets(data, char_offset, byte_to_char)
            )

        while position < len(text):
            matched = next(
                (token for token in special if text.startswith(token, position)), None
            )
            if matched is None:
                position += 1
                continue
            append_ordinary(text[ordinary_start:position], ordinary_start)
            output.append(
                (
                    self.added_tokens_encoder[matched],
                    (position, position + len(matched)),
                )
            )
            position += len(matched)
            ordinary_start = position
        append_ordinary(text[ordinary_start:], ordinary_start)
        return output

    def _tokenize(self, text, **kwargs):
        del kwargs
        return [self.decoder[index] for index in self._encode_bytes(text.encode("utf-8"))]

    def __call__(self, text=None, text_pair=None, **kwargs):
        output = super().__call__(text=text, text_pair=text_pair, **kwargs)
        if text_pair is not None or text is None:
            return output
        texts = [text] if isinstance(text, str) else list(text)
        if any(not isinstance(item, str) for item in texts):
            return output
        input_ids = output["input_ids"]
        attention_mask = output.get("attention_mask")
        raw_ids = input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids
        if raw_ids and isinstance(raw_ids[0], int):
            rows = [raw_ids]
            masks = [attention_mask] if attention_mask is not None else None
        else:
            rows = raw_ids
            if attention_mask is None:
                masks = None
            else:
                masks = (
                    attention_mask.tolist()
                    if hasattr(attention_mask, "tolist")
                    else attention_mask
                )
        encodings = []
        for index, source in enumerate(texts):
            exact = self._encode_with_offsets(source)
            row = rows[index]
            active = len(row) if masks is None else sum(int(item) for item in masks[index])
            exact = exact[:active]
            offsets = [offset for _token_id, offset in exact]
            missing = len(row) - len(offsets)
            if self.padding_side == "left":
                offsets = [None] * missing + offsets
            else:
                offsets.extend([None] * missing)
            encodings.append(_CharEncoding(offsets))
        output._encodings = encodings
        return output

    def _convert_token_to_id(self, token):
        return self.encoder.get(token, 0)

    def _convert_id_to_token(self, index):
        return self.decoder.get(index, END_TOKEN)

    def convert_tokens_to_string(self, tokens):
        decoded = bytearray()
        for token in tokens:
            raw = self._token_bytes.get(token)
            if raw is not None:
                decoded.extend(raw)
            else:
                decoded.extend(str(token).encode("utf-8"))
        return bytes(decoded).decode("utf-8", errors="replace")

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos = [self.bos_token_id] if self.add_bos_token else []
        output = bos + list(token_ids_0)
        if token_ids_1 is not None:
            output.extend(bos + list(token_ids_1))
        return output

    def save_vocabulary(self, save_directory, filename_prefix: Optional[str] = None):
        if not os.path.isdir(save_directory):
            raise ValueError("save directory does not exist")
        filename = (f"{filename_prefix}-" if filename_prefix else "") + "tokenizer.json"
        destination = os.path.join(save_directory, filename)
        if os.path.abspath(destination) != os.path.abspath(self.tokenizer_file):
            shutil.copyfile(self.tokenizer_file, destination)
        return (destination,)


__all__ = ["Rwkv7Tokenizer"]
