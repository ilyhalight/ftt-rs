"""Generate a small, strongly peaked softmax model for norm-skip benchmarks.

The first 16 labels have large-norm rows; the other 2,086 have tiny rows. A
hidden vector therefore has a high score against the big labels and provably
negligible scores against the rest, so Cauchy-Schwarz block bounds can skip
almost all output work.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

DIM = 64
NWORDS = 5_000
NLABELS = 2_102
BUCKET = 20_000
BIG = 16
OUT = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else Path(__file__).resolve().parent.parent / "bench" / "peaked.bin"
)
LINES = OUT.with_suffix(".txt")
rng = np.random.default_rng(123)


def matrix(f, rows: int, scales: np.ndarray | float) -> None:
    f.write(struct.pack("<qq", rows, DIM))
    chunk = 10_000
    for start in range(0, rows, chunk):
        n = min(chunk, rows - start)
        if np.isscalar(scales):
            scale = np.full(n, float(scales), dtype=np.float32)
        else:
            scale = scales[start : start + n]
        f.write((rng.standard_normal((n, DIM), dtype=np.float32) * scale[:, None]).tobytes())


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "wb") as f:
        f.write(struct.pack("<ii", 793_712_314, 12))
        f.write(
            struct.pack(
                "<12id",
                DIM, 5, 1, 1, 5, 1, 3, 3, BUCKET, 2, 4, 100, 1e-4,
            )
        )
        size = NWORDS + NLABELS
        f.write(struct.pack("<iiiqq", size, NWORDS, NLABELS, 10_000_000, -1))
        for i in range(NWORDS):
            word = b"</s>" if i == 0 else f"w{i}".encode()
            f.write(word + b"\0")
            f.write(struct.pack("<qb", 1_000_000 - i, 0))
        for i in range(NLABELS):
            f.write(f"__label__L{i}".encode() + b"\0")
            f.write(struct.pack("<qb", 1000 - (i % 1000), 1))

        f.write(b"\0")
        matrix(f, NWORDS + BUCKET, 1.0)
        f.write(b"\0")
        scales = np.full(NLABELS, 0.02, dtype=np.float32)
        scales[:BIG] = 20.0
        matrix(f, NLABELS, scales)

    lines = []
    for _ in range(20_000):
        ids = rng.integers(1, NWORDS, size=20)
        lines.append(" ".join(f"w{i}" for i in ids))
    LINES.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.0f} MB) and {LINES}")


if __name__ == "__main__":
    main()
