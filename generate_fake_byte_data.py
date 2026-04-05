"""Generate tiny fake byte-level shards for local smoke testing.

Creates data/datasets/fineweb10B_byte260/ with:
  - fineweb_train_000000.bin  (1M tokens)
  - fineweb_val_000000.bin    (200K tokens)

Token IDs are random bytes in range [4, 259] (the pure byte range).
Also writes a minimal fineweb_pure_byte_260.json tokenizer config.

Usage:
    python3 generate_fake_byte_data.py
"""

import json
import numpy as np
from pathlib import Path

MAGIC = 20240520
VERSION = 1
BYTE_OFFSET = 4
BYTE_COUNT = 256
VOCAB_SIZE = BYTE_OFFSET + BYTE_COUNT  # 260

TRAIN_TOKENS = 1_000_000
VAL_TOKENS   = 200_000


def write_shard(path: Path, num_tokens: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = np.zeros(256, dtype="<i4")
    header[0] = MAGIC
    header[1] = VERSION
    header[2] = num_tokens
    # random byte tokens in [BYTE_OFFSET, VOCAB_SIZE)
    tokens = np.random.randint(BYTE_OFFSET, VOCAB_SIZE, size=num_tokens, dtype=np.uint16)
    with path.open("wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype("<u2").tobytes())
    print(f"wrote {path}  ({num_tokens:,} tokens, {path.stat().st_size:,} bytes)")


def write_tokenizer_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "tokenizer_type": "pure_byte",
        "config": {
            "pad_id": 0,
            "bos_id": 1,
            "eos_id": 2,
            "unk_id": 3,
            "byte_offset": BYTE_OFFSET,
            "byte_count": BYTE_COUNT,
        },
        "vocab_size": VOCAB_SIZE,
    }
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    np.random.seed(42)
    base = Path("data/datasets/fineweb10B_byte260")
    write_shard(base / "fineweb_train_000000.bin", TRAIN_TOKENS)
    write_shard(base / "fineweb_val_000000.bin",   VAL_TOKENS)
    write_tokenizer_config(Path("data/tokenizers/fineweb_pure_byte_260.json"))
    print("\nDone. Run smoke test with:")
    print("RUN_ID=byte_smoke ITERATIONS=50 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=32768 python3 train_gpt_mlx_experimental.py")
