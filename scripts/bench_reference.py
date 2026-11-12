"""Time the reference fastText Python package on bench/lines.txt."""

import os
import sys
import tempfile
import time
from pathlib import Path

import fasttext

ROOT = Path(__file__).resolve().parent.parent
LINES = Path(os.environ.get("BENCH_LINES", ROOT / "bench" / "lines.txt"))
MODEL = os.environ.get(
    "FT_MODEL", str(Path(tempfile.gettempdir()) / "ftmodels" / "lid.176.ftz")
)
LIMIT = int(os.environ.get("BENCH_N", "0")) or None


def main() -> None:
    model = fasttext.load_model(MODEL)
    lines = LINES.read_text(encoding="utf-8").splitlines()[:LIMIT]
    print(f"model {MODEL}, {len(lines)} lines, fasttext {fasttext.__file__}")
    for threads in (1,):
        # Stock fastText has no threads parameter; this loop is serial.
        start = time.perf_counter()
        labels, probs = model.f.multilinePredict(lines, 1, 0.0, "strict")
        elapsed = time.perf_counter() - start
        print(f"multilinePredict: {elapsed:.3f} s, {len(lines) / elapsed:,.0f} lines/s")
        assert len(labels) == len(lines)
    # One-by-one path (what most Python callers use).
    start = time.perf_counter()
    for line in lines:
        model.f.predict(line + "\n", 1, 0.0, "strict")
    elapsed = time.perf_counter() - start
    print(f"predict one by one: {elapsed:.3f} s, {len(lines) / elapsed:,.0f} lines/s")


if __name__ == "__main__":
    main()
