"""Generate a 20k-line multilingual benchmark input (no model needed)."""

import random
from pathlib import Path

SEEDS = [
    "Hello, how are you today? I hope you are doing well and enjoying the weather.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "This product is absolutely wonderful and I would recommend it to everyone.",
    "I am very disappointed with the service and would like a refund immediately.",
    "Bonjour tout le monde, comment allez-vous aujourd'hui ?",
    "Je trouve ce film excellent, les acteurs sont formidables.",
    "Ce produit est terrible, je ne le recommande pas du tout.",
    "Привет, как дела? Сегодня прекрасная погода, хочется гулять.",
    "Мне очень понравился этот фильм, рекомендую всем посмотреть.",
    "Это ужасный сервис, я очень разочарован и хочу вернуть деньги.",
    "你好，今天过得怎么样？天气很好，我们一起去公园吧。",
    "这个产品非常棒，我非常喜欢，推荐给大家。",
    "这个服务太差了，我很生气，要求退款。",
    "こんにちは世界。今日はとても良い天気ですね。",
    "この映画は本当に素晴らしかったです。",
    "مرحبا كيف حالك اليوم؟ الطقس جميل جدا.",
    "أحب هذا المنتج كثيرا وأنصح به الجميع.",
    "Γειά σου κόσμε, τι κάνεις σήμερα;",
    "Olá mundo, tudo bem com você hoje?",
    "Este produto é maravilhoso, recomendo a todos.",
    "Hola mundo, ¿cómo estás hoy?",
    "Este servicio es terrible y estoy muy decepcionado.",
    "Hallo Welt, wie geht es dir heute?",
    "Dieses Produkt ist fantastisch und ich kann es nur empfehlen.",
    "Ciao mondo, come stai oggi?",
    "Questo film è davvero fantastico, lo consiglio a tutti.",
    "Merhaba dünya, bugün nasılsın?",
    "Ten produkt jest wspaniały i polecam go wszystkim.",
    "Dit product is geweldig en ik raad het iedereen aan.",
    "Det här är en mycket bra film som jag rekommenderar.",
    "🙂🚀 naïve café résumé — testing punctuation, numbers 12345 and symbols!",
    "random tokens zzqxv qwertyuiop asdfghjkl",
]

rng = random.Random(2026)
out = Path(__file__).resolve().parent.parent / "bench" / "lines.txt"
out.parent.mkdir(exist_ok=True)
lines = []
for _ in range(20_000):
    n = rng.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
    lines.append(" ".join(rng.choice(SEEDS) for _ in range(n)))
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {len(lines)} lines to {out} ({out.stat().st_size / 1e6:.1f} MB)")
