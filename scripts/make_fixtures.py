"""Train tiny fastText models and record reference predictions.

Run with a Python that has the reference `fasttext` package installed:
    py -3.11 scripts/make_fixtures.py

The tiny models and their expected outputs are written to tests/data/ so the
Rust test-suite does not need Python or network access.
"""

from __future__ import annotations

import json
import random
import struct
from pathlib import Path

import fasttext

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "tests" / "data"
MODELS = DATA / "models"

POS = [
    "good great excellent amazing wonderful fantastic superb delightful",
    "love like enjoy happy pleased glad thrilled satisfied impressed",
    "best awesome brilliant perfect outstanding marvelous terrific",
    "рекомендую отлично прекрасно замечательно здорово класс",
    "好 很好 优秀 喜欢 开心 满意 完美",
    "جيد رائع ممتاز جميل سعيد",
    "bueno excelente genial feliz maravilloso",
    "bon bon excellent génial heureux",
]
NEG = [
    "bad terrible awful horrible dreadful poor disappointing",
    "hate dislike sad angry upset annoyed frustrated miserable",
    "worst broken useless boring trash garbage",
    "плохо ужасно отвратительно грустно злой",
    "差 很糟 讨厌 伤心 生气 垃圾",
    "سيء فظيع حزين غاضب",
    "malo terrible horrible triste enojado",
    "mauvais terrible horrible triste en colère",
]
FILLER = [
    "the a of to and in is it this that with for on",
    "movie film product service app story book game",
    "today yesterday evening morning week month year",
    "i you we they he she my your our their",
    "really very quite so just too much little",
]

LABELS = ["__label__pos", "__label__neg", "__label__other"]


def make_corpus(path: Path, n: int = 600, seed: int = 7) -> list[str]:
    rng = random.Random(seed)
    lines = []
    for i in range(n):
        label = LABELS[i % 3]
        pool = POS if label == LABELS[0] else NEG if label == LABELS[1] else POS + NEG
        parts = [rng.choice(pool), rng.choice(pool), rng.choice(FILLER), rng.choice(FILLER)]
        rng.shuffle(parts)
        # Sprinkle in rare and out-of-vocabulary tokens, punctuation, and unicode.
        if i % 11 == 0:
            parts.append("zzqxv" + str(i))
        if i % 13 == 0:
            parts.append("naïve-café_42")
        if i % 17 == 0:
            parts.append("🙂🚀")
        if i % 19 == 0:
            parts.append("__label__pos")  # also appears as a feature in training
        lines.append(label + " " + " ".join(parts))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


TEST_LINES = [
    "good great excellent movie",
    "terrible awful boring product",
    "I really love this wonderful app",
    "I hate this broken useless garbage",
    "отлично прекрасно рекомендую",
    "плохо ужасно грустно",
    "好 很好 优秀 喜欢",
    "差 很糟 讨厌",
    "bueno excelente genial",
    "bon excellent génial",
    "zzqxv999 unseenword qwertyuiop",
    "🙂 🚀 naïve café",
    "__label__pos hello",
    "the a of to and in",
    "   ",
    "",
    "\tgood\tgreat\r",
    
    "a" * 300,
    "good " * 40,
]


def predicted(model, line: str, k: int, threshold: float = 0.0):
    preds = model.f.predict(line + "\n", k, threshold, "strict")
    probs = [p for p, _ in preds]
    labels = [lab for _, lab in preds]
    return {
        "labels": [str(x) for x in labels],
        "probs": [float(x) for x in probs],
    }


FIXTURE_MAGIC = 0x46544631  # "FTF1"


def dump_split(name: str, model, lines: list[str], specs: dict) -> None:
    out = {}
    for spec_name, (k, threshold) in specs.items():
        out[spec_name] = [predicted(model, line, k, threshold) for line in lines]
    (DATA / f"expected_{name}.json").write_text(
        json.dumps({"lines": lines, "cases": out}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    # Compact, dependency-free fixture read by tests/compat.rs.
    with open(DATA / f"fixtures_{name}.ftf", "wb") as f:
        f.write(struct.pack("<II", FIXTURE_MAGIC, len(lines)))
        for line in lines:
            b = line.encode("utf-8")
            f.write(struct.pack("<I", len(b)))
            f.write(b)
        f.write(struct.pack("<I", len(specs)))
        for spec_name, (k, threshold) in specs.items():
            f.write(struct.pack("<IfI", k, threshold, len(lines)))
            for pred in out[spec_name]:
                labels = pred["labels"]
                probs = pred["probs"]
                assert len(labels) == len(probs)
                f.write(struct.pack("<I", len(labels)))
                for label, prob in zip(labels, probs):
                    b = label.encode("utf-8")
                    f.write(struct.pack("<fI", prob, len(b)))
                    f.write(b)


def write_peaked_model(path: Path) -> list[str]:
    """A tiny hand-written softmax model whose first four label rows have much
    larger norms than the rest, so norm-bound skipping is guaranteed to fire.
    """
    rng = random.Random(123)
    dim, nwords, nlabels, bucket = 16, 64, 128, 1000
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", 793_712_314, 12))
        f.write(
            struct.pack(
                "<12id", dim, 5, 1, 1, 5, 1, 3, 3, bucket, 0, 0, 100, 1e-4
            )
        )
        f.write(
            struct.pack(
                "<iiiqq", nwords + nlabels, nwords, nlabels, 1_000_000, -1
            )
        )
        words = ["</s>"] + [f"w{i}" for i in range(1, nwords)]
        for word in words:
            f.write(word.encode() + b"\0")
            f.write(struct.pack("<qb", 1000, 0))
        for i in range(nlabels):
            f.write(f"__label__L{i}".encode() + b"\0")
            f.write(struct.pack("<qb", 1000, 1))

        f.write(b"\0")
        f.write(struct.pack("<qq", nwords + bucket, dim))
        for _ in range(nwords + bucket):
            f.write(struct.pack(f"<{dim}f", *[rng.gauss(0, 1) for _ in range(dim)]))
        f.write(b"\0")
        f.write(struct.pack("<qq", nlabels, dim))
        for row in range(nlabels):
            scale = 20.0 if row < 4 else 0.01
            values = [rng.gauss(0, 1) * scale for _ in range(dim)]
            f.write(struct.pack(f"<{dim}f", *values))

    return [
        "w1 w2 w3",
        "w10 w11 w12 w13",
        "w40 w41",
        "w20 w21 w22 w23 w24",
        "hello world",
        "w5 " * 50,
        "",
        "w63 w62",
        "w0 w1",
        "w30 w31 w32",
    ]


def train(name: str, corpus: Path, **kwargs):
    model = fasttext.train_supervised(
        input=str(corpus), thread=1, seed=42, verbose=0, **kwargs
    )
    path = MODELS / f"{name}.bin"
    model.save_model(str(path))
    return model


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    corpus = DATA / "corpus.txt"
    make_corpus(corpus)
    specs = {"top5": (5, 0.0), "top1": (1, 0.0), "top3_threshold": (3, 0.02)}

    configs = {
        "dense_softmax": dict(dim=16, epoch=8, minn=2, maxn=4, bucket=2000,
                              wordNgrams=1, loss="softmax", minCount=1),
        "dense_wngrams": dict(dim=13, epoch=6, minn=3, maxn=5, bucket=777,
                              wordNgrams=3, loss="softmax", minCount=1),
        "dense_hs": dict(dim=12, epoch=6, minn=1, maxn=3, bucket=333,
                         wordNgrams=2, loss="hs", minCount=1),
        "dense_ova": dict(dim=8, epoch=6, minn=0, maxn=0, bucket=0,
                          wordNgrams=1, loss="ova", minCount=1),
        "dense_nonchar": dict(dim=10, epoch=5, minn=0, maxn=0, bucket=0,
                              wordNgrams=1, loss="softmax", minCount=1),
        "dense_ns": dict(dim=9, epoch=6, minn=2, maxn=4, bucket=500,
                         wordNgrams=2, loss="ns", neg=5, minCount=1),
    }
    for name, cfg in configs.items():
        model = train(name, corpus, **cfg)
        dump_split(name, model, TEST_LINES, specs)

    # A hand-written peaked softmax model: exercises norm-bound skipping.
    peaked_lines = write_peaked_model(MODELS / "peaked.bin")
    peaked = fasttext.load_model(str(MODELS / "peaked.bin"))
    dump_split("peaked", peaked, peaked_lines, {"top1": (1, 0.0)})

    # An unsupervised model: loaded for inspection, but not usable for
    # classification prediction.
    unsup = fasttext.train_unsupervised(
        input=str(corpus), dim=8, epoch=2, minn=2, maxn=3, bucket=1000,
        thread=1, verbose=0,
    )
    unsup.save_model(str(MODELS / "unsup.bin"))

    # Quantized models (input only, and input+output).
    for name, qout, qnorm, dsub in [
        ("quant_in", False, False, 2),
        ("quant_qnorm", False, True, 2),
        ("quant_dsub4", False, True, 4),
    ]:
        model = fasttext.load_model(str(MODELS / "dense_softmax.bin"))
        model.quantize(
            input=str(corpus), qout=qout, cutoff=0, retrain=False, qnorm=qnorm, dsub=dsub
        )
        model.save_model(str(MODELS / f"{name}.ftz"))
        dump_split(name, model, TEST_LINES, specs)

    # A model with enough labels to quantize the output matrix as well.
    many = DATA / "corpus_manylabels.txt"
    rows = []
    for label in range(300):
        for j in range(24):
            rows.append(
                f"__label__L{label} shared alpha beta group{label % 7} "
                f"w{label}_{j} w{label}_{(j + 1) % 24} token{label * 24 + j}"
            )
    many.write_text("\n".join(rows) + "\n", encoding="utf-8")
    model = train("manylabels", many, dim=16, epoch=3, minn=0, maxn=0,
                  bucket=0, wordNgrams=1, loss="softmax", minCount=1)
    model.quantize(input=str(many), qout=True, cutoff=0, retrain=False,
                   qnorm=True, dsub=2)
    model.save_model(str(MODELS / "quant_qout.ftz"))
    many_lines = [
        "shared alpha beta group3 w17_4 w17_5",
        "shared alpha beta group5 w299_1 w299_2",
        "w0_0 w1_1 w2_2",
        "",
    ]
    dump_split("quant_qout", model, many_lines, specs)

    # Quantized output with hierarchical softmax: the tree walks PQ rows.
    model = train("manylabels_hs", many, dim=16, epoch=3, minn=0, maxn=0,
                  bucket=0, wordNgrams=1, loss="hs", minCount=1)
    model.quantize(input=str(many), qout=True, cutoff=0, retrain=False,
                   qnorm=True, dsub=2)
    model.save_model(str(MODELS / "quant_qout_hs.ftz"))
    dump_split("quant_qout_hs", model, many_lines, specs)

    # These .bin files are only intermediates used to train the .ftz models.
    (MODELS / "manylabels.bin").unlink(missing_ok=True)
    (MODELS / "manylabels_hs.bin").unlink(missing_ok=True)

    # Keep only the fixture files, not copies of the corpus models used above.
    meta = {
        "test_lines": len(TEST_LINES),
        "corpus": "corpus.txt",
        "models": sorted(p.name for p in MODELS.iterdir()),
        "note": "Generated by scripts/make_fixtures.py with fasttext 0.9.x",
    }
    (DATA / "fixtures.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"wrote fixtures to {DATA}")


if __name__ == "__main__":
    main()
