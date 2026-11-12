"""Compare the Rust CLI against the reference fastText Python package.

Usage:
    py -3.11 scripts/compare_reference.py [--ftt target/release/ftt.exe]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import fasttext

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "tests" / "data"


def run_ftt(ftt: str, model: Path, lines: list[str], k: int, threads: int = 0):
    text = "\n".join(lines) + "\n"
    cmd = [ftt, "predict-prob", str(model), "-k", str(k), "-threads", str(threads)]
    out = subprocess.run(cmd, input=text.encode(), stdout=subprocess.PIPE, check=True)
    rows = []
    for line in out.stdout.decode().splitlines():
        toks = line.split()
        preds = [(toks[i], float(toks[i + 1])) for i in range(0, len(toks), 2)]
        rows.append(preds)
    assert len(rows) == len(lines), (len(rows), len(lines))
    return rows


def reference(model_path: Path, lines: list[str], k: int):
    model = fasttext.load_model(str(model_path))
    rows = []
    for line in lines:
        preds = model.f.predict(line + "\n", k, 0.0, "strict")
        rows.append([(lab, p) for p, lab in preds])
    return rows


def check(ftt: str, name: str, model: Path, lines: list[str], k: int = 5) -> tuple[int, float]:
    got = run_ftt(ftt, model, lines, k)
    want = reference(model, lines, k)
    bad = 0
    max_diff = 0.0
    for n, (g, w) in enumerate(zip(got, want)):
        gl = [x[0] for x in g]
        wl = [x[0] for x in w]
        if gl != wl:
            print(f"[{name}] line {n}: labels differ\n  got  {gl}\n  want {wl}\n  {lines[n]!r}")
            bad += 1
            continue
        for (_, gp), (_, wp) in zip(g, w):
            max_diff = max(max_diff, abs(gp - wp))
    print(f"[{name}] {len(lines)} lines, {bad} label mismatches, max |dp| = {max_diff:.3g}")
    return bad, max_diff


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ftt", default=str(ROOT / "target" / "release" / "ftt.exe"))
    args = ap.parse_args()

    failures = 0
    max_diff = 0.0
    for expected in sorted(DATA.glob("expected_*.json")):
        name = expected.stem.removeprefix("expected_")
        obj = json.loads(expected.read_text(encoding="utf-8"))
        lines = obj["lines"]
        model = DATA / "models" / (
            f"{name}.ftz" if (DATA / "models" / f"{name}.ftz").exists() else f"{name}.bin"
        )
        if not model.exists():
            continue
        ks = {"top5": 5, "top1": 1, "top3_threshold": 3}
        for spec in obj["cases"]:
            if spec not in ks:
                continue
            bad, md = check(args.ftt, f"{name}/{spec}", model, lines, ks[spec])
            failures += bad
            max_diff = max(max_diff, md)

    # The real GlotLID/lid model, if it was downloaded.
    lid = Path.home() / "AppData" / "Local" / "Temp" / "ftmodels" / "lid.176.ftz"
    if not lid.exists():
        lid = Path("/tmp/ftmodels/lid.176.ftz")
    if lid.exists():
        lines = [
            "Hello, how are you today?",
            "Bonjour tout le monde, comment allez-vous ?",
            "Привет, как дела?",
            "你好，今天过得怎么样？",
            "مرحبا كيف حالك",
            "Γειά σου κόσμε",
            "こんにちは世界",
            "안녕하세요 세계",
            "Olá mundo, tudo bem?",
            "random tokens zzqxv qwerty",
            "",
            "🙂🚀 naïve café",
        ]
        bad, md = check(args.ftt, "lid.176.ftz", lid, lines, 5)
        failures += bad
        max_diff = max(max_diff, md)

    print(f"total label mismatches: {failures}, overall max |dp| = {max_diff:.3g}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
