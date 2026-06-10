"""
Навчання ваг байєсівського агрегування стилометричного детектора
на корпусі з 800 текстів (training_corpus/corpus.csv).

Метод: логістична регресія на стандартизованих 9 ознаках
з L2-регуляризацією. Коефіцієнти β_i інтерпретуються як ваги
доказів (log-likelihood ratios) у наївно-байєсівській агрегації:

    log[ P(AI|x) / P(human|x) ] = β_0 + Σ β_i · z_i

де z_i = (x_i − μ_i) / σ_i — стандартизована ознака.

Валідація: 5-кратна стратифікована перехресна перевірка з
обчисленням AUC ROC, F1, Precision, Recall.
"""

import csv
import json
import os
from datetime import datetime, timezone

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from stylometric import FEATURE_NAMES, extract_features


SEED = 42
CORPUS_DIR = "training_corpus"
N_FOLDS = 5
L2_C = 1.0  # 1/λ, менше значення = сильніша регуляризація


def load_corpus():
    csv_path = os.path.join(CORPUS_DIR, "corpus.csv")
    if not os.path.exists(csv_path):
        raise SystemExit(
            f"Не знайдено {csv_path}. Спочатку запусти: python build_corpus.py"
        )
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            with open(os.path.join(CORPUS_DIR, "texts", r["filename"]),
                      encoding="utf-8") as tf:
                r["text"] = tf.read()
            rows.append(r)
    return rows


def main():
    print("Завантаження корпусу...")
    rows = load_corpus()
    print(f"  {len(rows)} текстів")

    print("Витяг ознак (9 стилометричних)...")
    X = np.array([extract_features(r["text"]) for r in rows], dtype=float)
    y = np.array([1 if r["label"] == "ai" else 0 for r in rows], dtype=int)
    print(f"  Матриця: {X.shape}\n")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv_metrics = {"auc": [], "f1": [], "precision": [], "recall": []}
    fold_threshold = []

    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        clf = LogisticRegression(
            C=L2_C, max_iter=2000, random_state=SEED, solver="lbfgs",
        ).fit(Xtr, y[tr])
        proba = clf.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        cv_metrics["auc"].append(roc_auc_score(y[te], proba))
        cv_metrics["f1"].append(f1_score(y[te], pred))
        cv_metrics["precision"].append(precision_score(y[te], pred))
        cv_metrics["recall"].append(recall_score(y[te], pred))
        fpr, tpr, thr = roc_curve(y[te], proba)
        j = np.argmax(tpr - fpr)
        fold_threshold.append(float(thr[j]) if j < len(thr) else 0.5)
        print(f"  fold {fold}: AUC={cv_metrics['auc'][-1]:.4f}, "
              f"F1={cv_metrics['f1'][-1]:.4f}")

    print("\nФінальне навчання на повному корпусі...")
    scaler_full = StandardScaler().fit(X)
    X_scaled = scaler_full.transform(X)
    final_clf = LogisticRegression(
        C=L2_C, max_iter=2000, random_state=SEED, solver="lbfgs",
    ).fit(X_scaled, y)

    coefs = final_clf.coef_[0].tolist()
    intercept = float(final_clf.intercept_[0])
    means = scaler_full.mean_.tolist()
    stds = scaler_full.scale_.tolist()

    # Per-feature individual discrimination (AUC for each feature alone)
    feature_aucs = []
    for i in range(X.shape[1]):
        # one-feature logistic regression
        clf_i = LogisticRegression(max_iter=2000, random_state=SEED)
        clf_i.fit(X[:, i:i + 1], y)
        proba_i = clf_i.predict_proba(X[:, i:i + 1])[:, 1]
        feature_aucs.append(float(roc_auc_score(y, proba_i)))

    # Build features description
    features_desc = []
    for i, name in enumerate(FEATURE_NAMES):
        features_desc.append({
            "name": name,
            "weight": round(coefs[i], 6),
            "mean": round(means[i], 6),
            "std": round(stds[i], 6),
            "individual_auc": round(feature_aucs[i], 4),
            "direction": "AI-indicator" if coefs[i] > 0 else "human-indicator",
        })

    # Load manifest for documentation
    manifest_path = os.path.join(CORPUS_DIR, "MANIFEST.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

    weights = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "method": "Logistic regression (L2, C=1.0) on standardized features",
        "interpretation": (
            "log[P(AI|x)/P(human|x)] = intercept + Σ weight_i · "
            "(feature_i - mean_i) / std_i"
        ),
        "corpus": {
            "size": len(rows),
            "labels": {"human": int((y == 0).sum()), "ai": int((y == 1).sum())},
            "language_distribution": manifest.get("language_distribution", {}),
            "source": manifest.get("source_dataset", "ai-text-detection-pile"),
            "sha256": manifest.get("corpus_sha256", ""),
        },
        "intercept": round(intercept, 6),
        "features": features_desc,
        "cross_validation": {
            "folds": N_FOLDS,
            "auc_roc_mean": round(float(np.mean(cv_metrics["auc"])), 4),
            "auc_roc_std": round(float(np.std(cv_metrics["auc"])), 4),
            "f1_mean": round(float(np.mean(cv_metrics["f1"])), 4),
            "precision_mean": round(float(np.mean(cv_metrics["precision"])), 4),
            "recall_mean": round(float(np.mean(cv_metrics["recall"])), 4),
            "optimal_threshold_mean": round(float(np.mean(fold_threshold)), 4),
        },
    }

    weights_path = os.path.join(CORPUS_DIR, "weights.json")
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(weights, f, ensure_ascii=False, indent=2)

    # Human-readable report
    lines = []
    lines.append("=" * 70)
    lines.append("ЗВІТ ПРО НАВЧАННЯ ВАГ БАЙЄСІВСЬКОГО АГРЕГУВАННЯ")
    lines.append("=" * 70)
    lines.append(f"Дата:       {weights['trained_at']}")
    lines.append(f"Корпус:     {weights['corpus']['size']} текстів "
                 f"({weights['corpus']['labels']['human']} людських + "
                 f"{weights['corpus']['labels']['ai']} ШІ)")
    lines.append(f"Розподіл за мовами: {weights['corpus']['language_distribution']}")
    lines.append(f"SHA-256:    {weights['corpus']['sha256'][:32]}...")
    lines.append("")
    lines.append("Метод: " + weights["method"])
    lines.append("")
    lines.append("МЕТРИКИ 5-КРАТНОЇ ПЕРЕХРЕСНОЇ ПЕРЕВІРКИ")
    lines.append("-" * 70)
    cv = weights["cross_validation"]
    lines.append(f"  AUC ROC:    {cv['auc_roc_mean']:.4f} ± {cv['auc_roc_std']:.4f}")
    lines.append(f"  F1:         {cv['f1_mean']:.4f}")
    lines.append(f"  Precision:  {cv['precision_mean']:.4f}")
    lines.append(f"  Recall:     {cv['recall_mean']:.4f}")
    lines.append(f"  Поріг (Youden's J): {cv['optimal_threshold_mean']:.4f}")
    lines.append("")
    lines.append("ВАГИ ОЗНАК (β-коефіцієнти на стандартизованих ознаках)")
    lines.append("-" * 70)
    lines.append(f"{'Ознака':<22} {'Вага':>10} {'Напрям':<18} {'Окрема AUC':>10}")
    for f in features_desc:
        lines.append(f"{f['name']:<22} {f['weight']:>+10.4f} "
                     f"{f['direction']:<18} {f['individual_auc']:>10.4f}")
    lines.append(f"\n{'Перехоплення (β₀)':<22} {intercept:>+10.4f}")
    lines.append("")
    lines.append("Інтерпретація: log[P(AI)/P(human)] = β₀ + Σ βᵢ · zᵢ")
    lines.append(f"де zᵢ = (xᵢ − μᵢ) / σᵢ, параметри стандартизації у weights.json.")
    report_path = os.path.join(CORPUS_DIR, "training_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines))
    print(f"\n✓ Збережено: {weights_path}")
    print(f"✓ Збережено: {report_path}")


if __name__ == "__main__":
    main()
