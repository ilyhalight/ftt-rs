"""Write a valid random fastText model with GlotLID-like shape.

This is for benchmarks only: 2,102 labels, dimension 256, 20,000 words, and a
200,000-bucket n-gram table (~225 MB). Predictions are meaningless, but the
model exercises the same code paths as a real large-label classifier and can
be loaded by both fastText and fasttext-turbo.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

DIM = 256
NWORDS = 20_000
NLABELS = 2_102
NOGRAM = "--nogram" in sys.argv
BUCKET = 0 if NOGRAM else 200_000
MINN, MAXN = (0, 0) if NOGRAM else (2, 4)
OUT = Path(
    next(
        (a for a in sys.argv[1:] if not a.startswith("--")),
        Path(__file__).resolve().parent.parent / "bench" / ("nogram.bin" if NOGRAM else "synthetic.bin"),
    )
)
LINES = OUT.with_suffix(".txt")
rng = np.random.default_rng(2026)


def write_matrix(f, rows: int, scale: float) -> None:
    f.write(struct.pack("<qq", rows, DIM))
    chunk = 20_000
    for start in range(0, rows, chunk):
        n = min(chunk, rows - start)
        f.write((rng.standard_normal((n, DIM), dtype=np.float32) * scale).tobytes())


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "wb") as f:
        f.write(struct.pack("<ii", 793_712_314, 12))
        f.write(
            struct.pack(
                "<12id",
                DIM,      # dim
                5,        # ws
                1,        # epoch
                1,        # minCount
                5,        # neg
                1,        # wordNgrams
                3,        # loss: softmax
                3,        # model: sup
                BUCKET,
                MINN,
                MAXN,
                100,      # lrUpdateRate
                1e-4,     # t
            )
        )
        size = NWORDS + NLABELS
        f.write(struct.pack("<iiiqq", size, NWORDS, NLABELS, 10_000_000, -1))

        # Words (id 0 is </s>, as in real models).
        for i in range(NWORDS):
            word = b"</s>" if i == 0 else f"w{i}".encode()
            f.write(word + b"\0")
            f.write(struct.pack("<qb", 1_000_000 - i, 0))
        for i in range(NLABELS):
            word = f"__label__L{i}".encode()
            f.write(word + b"\0")
            f.write(struct.pack("<qb", 1000 - (i % 1000), 1))

        f.write(b"\0")  # quant_input = false
        write_matrix(f, NWORDS + BUCKET, 1.0)  # input rows
        f.write(b"\0")  # qout = false
        write_matrix(f, NLABELS, 1.0)  # output rows

    # 20 words per line -> roughly 650 character n-grams with minn=2/maxn=4.
    lines = []
    for _ in range(20_000):
        ids = rng.integers(1, NWORDS, size=20)
        lines.append(" ".join(f"w{i}" for i in ids))
    LINES.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.0f} MB) and {LINES}")


if __name__ == "__main__":
    main()
