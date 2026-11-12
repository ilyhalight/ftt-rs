"""Compare fasttext-turbo against reference fastText on a larger input file.

    py -3.11 scripts/compare_bulk.py <model> <input.txt> [n] [k]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import fasttext

ROOT = Path(__file__).resolve().parent.parent
FTT = ROOT / "target" / "release" / ("ftt.exe" if sys.platform == "win32" else "ftt")


def main() -> int:
    model_path = Path(sys.argv[1])
    input_path = Path(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    lines = input_path.read_text(encoding="utf-8").splitlines()[:n]

    model = fasttext.load_model(str(model_path))
    expected = []
    for line in lines:
        preds = model.f.predict(line + "\n", k, 0.0, "strict")
        expected.append(([lab for _, lab in preds], [p for p, _ in preds]))

    out = subprocess.run(
        [
            str(FTT),
            "predict-prob",
            str(model_path),
            "-k",
            str(k),
            "-threads",
            "8",
        ],
        input=("\n".join(lines) + "\n").encode(),
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.decode()
    rows = out.splitlines()
    assert len(rows) == len(lines), (len(rows), len(lines))

    label_bad = 0
    max_diff = 0.0
    for i, (row, (want_labels, want_probs)) in enumerate(zip(rows, expected)):
        toks = row.split()
        got_labels = toks[0::2]
        got_probs = [float(x) for x in toks[1::2]]
        if got_labels != want_labels:
            label_bad += 1
            if label_bad <= 5:
                print(f"line {i}: labels\n  got  {got_labels}\n  want {want_labels}")
        for a, b in zip(got_probs, want_probs):
            max_diff = max(max_diff, abs(a - b))
    print(
        f"{model_path.name}: {len(lines)} lines, {label_bad} label mismatches, "
        f"max |dp| = {max_diff:.3g}"
    )
    return 1 if label_bad else 0


if __name__ == "__main__":
    sys.exit(main())
