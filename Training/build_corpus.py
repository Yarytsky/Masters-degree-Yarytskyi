"""
Збір збалансованого навчального корпусу з 800 текстів.

Джерело: ai-text-detection-pile (HuggingFace, artem9k/ai-text-detection-pile).
Композиція: 400 людських + 400 згенерованих ШІ.
Фільтрація: мінімум 100 слів, максимум 2000 слів на текст.

"""

import csv
import hashlib
import json
import os
from datetime import datetime, timezone

from datasets import load_dataset

from stylometric import detect_lang


SAMPLE_PER_CLASS = 400
MIN_WORDS = 100
MAX_WORDS = 2000
SEED = 42
OUTPUT_DIR = "training_corpus"


def looks_ai(label):
    if isinstance(label, str):
        l = label.lower()
        return any(k in l for k in ("gpt", "ai", "claude", "llama", "synthetic"))
    if isinstance(label, (int, float)):
        return int(label) == 1
    return False


def fingerprint(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "texts"), exist_ok=True)

    print("Завантаження ai-text-detection-pile (streaming)...")
    ds = load_dataset(
        "artem9k/ai-text-detection-pile", split="train", streaming=True,
    ).shuffle(seed=SEED, buffer_size=10000)

    human, ai = [], []
    scanned = 0
    for row in ds:
        scanned += 1
        text = row.get("text") or row.get("content") or ""
        label = row.get("source", row.get("label", row.get("ai", None)))
        wc = len(text.split())
        if wc < MIN_WORDS or wc > MAX_WORDS:
            continue
        record = {
            "text": text,
            "source": str(label) if label is not None else "",
            "word_count": wc,
            "language": detect_lang(text),
            "hash": fingerprint(text),
        }
        if looks_ai(label) and len(ai) < SAMPLE_PER_CLASS:
            ai.append(record)
        elif not looks_ai(label) and len(human) < SAMPLE_PER_CLASS:
            human.append(record)
        if len(ai) >= SAMPLE_PER_CLASS and len(human) >= SAMPLE_PER_CLASS:
            break
        if scanned % 5000 == 0:
            print(f"  переглянуто {scanned}, зібрано: "
                  f"людських {len(human)}, ШІ {len(ai)}")

    print(f"\nЗібрано: людських {len(human)}, ШІ {len(ai)} (з {scanned} переглянутих)")
    if len(human) < SAMPLE_PER_CLASS or len(ai) < SAMPLE_PER_CLASS:
        raise SystemExit("Не вдалося набрати потрібну кількість зразків")

    rows = []
    for i, rec in enumerate(human + ai, start=1):
        label = "human" if i <= SAMPLE_PER_CLASS else "ai"
        text_id = f"{i:04d}"
        fname = f"text_{text_id}.txt"
        with open(os.path.join(OUTPUT_DIR, "texts", fname), "w", encoding="utf-8") as f:
            f.write(rec["text"])
        rows.append({
            "id": text_id,
            "label": label,
            "source": rec["source"],
            "language": rec["language"],
            "word_count": rec["word_count"],
            "hash": rec["hash"],
            "filename": fname,
        })

    csv_path = os.path.join(OUTPUT_DIR, "corpus.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lang_dist = {}
    for r in rows:
        lang_dist[r["language"]] = lang_dist.get(r["language"], 0) + 1

    corpus_hash = hashlib.sha256()
    for r in sorted(rows, key=lambda x: x["id"]):
        corpus_hash.update(r["hash"].encode("utf-8"))

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "size": len(rows),
        "labels": {"human": SAMPLE_PER_CLASS, "ai": SAMPLE_PER_CLASS},
        "language_distribution": lang_dist,
        "source_dataset": "artem9k/ai-text-detection-pile",
        "filter": {
            "min_words": MIN_WORDS,
            "max_words": MAX_WORDS,
            "seed": SEED,
        },
        "scanned_rows": scanned,
        "corpus_sha256": corpus_hash.hexdigest(),
    }
    with open(os.path.join(OUTPUT_DIR, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nЗбережено у {OUTPUT_DIR}/:")
    print(f"  corpus.csv         ({len(rows)} рядків)")
    print(f"  texts/             ({len(rows)} файлів)")
    print(f"  MANIFEST.json      (sha256: {manifest['corpus_sha256'][:16]}...)")
    print(f"\nМовний розподіл: {lang_dist}")


if __name__ == "__main__":
    main()
