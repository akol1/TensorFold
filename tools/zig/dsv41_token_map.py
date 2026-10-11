#!/usr/bin/env python3
"""Offline Engram metadata packing; official algorithm and MIT attribution in THIRD_PARTY_NOTICES.md."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np
from tokenizers import Regex, Tokenizer, normalizers

SOURCE = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/inference/engram.py"
MAGIC = b"DSENGR01"
RAW_VOCAB = 129280
COMPRESSED_VOCAB = 99092


def compressed_map(backend: Tokenizer) -> tuple[list[int], int]:
    sentinel = "\ue000"
    normalize = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(),
        normalizers.Lowercase(), normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel), normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])
    keys: dict[str, int] = {}
    lookup: list[int] = []
    if backend.get_vocab_size(with_added_tokens=True) != RAW_VOCAB:
        raise ValueError("This native format supports the released vocabulary only")
    for token_id in range(backend.get_vocab_size(with_added_tokens=True)):
        text = backend.decode([token_id], skip_special_tokens=False)
        key = backend.id_to_token(token_id) if "\ufffd" in text else normalize.normalize_str(text) or text
        if key is None:
            raise ValueError(f"Missing tokenizer entry {token_id}")
        if key not in keys:
            keys[key] = len(keys)
        lookup.append(keys[key])
    return lookup, len(keys)


def prime(n: int) -> bool:
    if n < 2 or n % 2 == 0:
        return n == 2
    return all(n % divisor for divisor in range(3, math.isqrt(n) + 1, 2))


def layout(config: dict, compressed: int) -> list[dict]:
    seen: set[int] = set()
    layers = []
    order, heads = config["engram_max_ngram_size"], config["engram_n_heads"]
    bound = max(1, ((2**63 - 1) // compressed) // 2)
    for layer_id, rows in zip(config["engram_layer_ids"], config["engram_num_embeddings"], strict=True):
        primes, offsets = [], []
        for _ in range(order - 1):
            current = config["engram_vocab_size"] - 1
            for _ in range(heads):
                current += 1
                while current in seen or not prime(current):
                    current += 1
                seen.add(current)
                offsets.append(sum(primes))
                primes.append(current)
        if sum(primes) != rows:
            raise ValueError(f"Layer {layer_id}: prime buckets sum {sum(primes)} != table rows {rows}")
        multipliers = np.random.default_rng(10007 * layer_id).integers(0, bound, size=order, dtype=np.int64) * 2 + 1
        layers.append(dict(layer_id=layer_id, rows=rows, primes=primes, offsets=offsets,
                           multipliers=[int(x) for x in multipliers]))
    return layers


def released_config(config: dict) -> None:
    expected = dict(vocab_size=RAW_VOCAB, engram_compressed_vocab_size=COMPRESSED_VOCAB,
                    engram_max_ngram_size=4, engram_n_heads=8, engram_vocab_size=16000000)
    for name, value in expected.items():
        if type(config.get(name)) is not int or config[name] != value:
            raise ValueError(f"Unsupported released configuration: {name}")
    if config.get("engram_layer_ids") != [1, 14]:
        raise ValueError("Unsupported Engram layers")
    rows = config.get("engram_num_embeddings")
    if not isinstance(rows, list) or len(rows) != 2 or any(
        type(row) is not int or not 0 < row <= 24 * 16010000 for row in rows
    ):
        raise ValueError("Invalid Engram table dimensions")


def bounded_bytes(path: Path, limit: int) -> bytes:
    with path.open("rb") as source:
        data = source.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"Metadata exceeds {limit} bytes")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="Pinned official inference/config.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_bytes = bounded_bytes(args.config, 1 << 20)
    outer = json.loads(config_bytes)
    config = outer.get("text_config", outer)
    released_config(config)
    tokenizer_bytes = bounded_bytes(args.tokenizer, 64 << 20)
    lookup, compressed = compressed_map(Tokenizer.from_str(tokenizer_bytes.decode("utf-8")))
    if compressed != config["engram_compressed_vocab_size"] or len(lookup) != config["vocab_size"]:
        raise ValueError(f"Tokenizer mismatch: raw={len(lookup)}, compressed={compressed}")
    order, heads = config["engram_max_ngram_size"], config["engram_n_heads"]
    pad_id = config.get("engram_pad_id", config.get("engram_pad_token_id"))
    if type(pad_id) is not int or not 0 <= pad_id < len(lookup):
        raise ValueError("Invalid Engram pad token")
    pad = lookup[pad_id]
    layers = layout(config, compressed)
    tokenizer_sha = hashlib.sha256(tokenizer_bytes).digest()
    config_sha = hashlib.sha256(config_bytes).digest()
    # LE header: magic, raw/compressed vocab, compressed pad, order, heads, layers, provenance hashes.
    packed = bytearray(MAGIC + struct.pack("<6I", len(lookup), compressed, pad, order, heads, len(layers)))
    packed.extend(tokenizer_sha + config_sha)
    for layer in layers:
        packed.extend(struct.pack("<2I4Q24I24I", layer["layer_id"], layer["rows"],
                                  *layer["multipliers"], *layer["primes"], *layer["offsets"]))
    packed.extend(struct.pack(f"<{len(lookup)}I", *lookup))
    args.output.write_bytes(packed)
    manifest = dict(source=SOURCE, tokenizer_sha256=tokenizer_sha.hex(), config_sha256=config_sha.hex(),
                    packed_sha256=hashlib.sha256(packed).hexdigest(), raw_vocab=len(lookup),
                    compressed_vocab=compressed, pad=pad, order=order, heads=heads, layers=layers)
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
