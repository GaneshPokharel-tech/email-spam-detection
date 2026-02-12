from __future__ import annotations

from pathlib import Path
import re
from datetime import timezone
from email.utils import parsedate_to_datetime

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


DATE_RE = re.compile(r"^Date:\s*(.+)$", re.IGNORECASE | re.MULTILINE)


def extract_date_utc(email_text: str):
    """Parse Date: header -> timezone-aware UTC datetime. Return None if missing/invalid."""
    m = DATE_RE.search(email_text)
    if not m:
        return None
    raw = m.group(1).strip()
    try:
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        # Filter weird years
        if dt.year < 1990 or dt.year > 2010:
            return None
        return dt
    except Exception:
        return None


def load_rows(data_dir: Path):
    """Load SpamAssassin folders and return list of dicts: text,label,date."""
    rows = []

    def load_folder(folder: Path, label: int):
        for p in sorted(folder.glob("*")):
            if not p.is_file():
                continue
            text = p.read_text(errors="ignore")
            dt = extract_date_utc(text)
            if dt is None:
                continue
            rows.append({"text": text, "label": label, "date": dt})

    load_folder(data_dir / "easy_ham", 0)
    load_folder(data_dir / "spam_2", 1)
    return rows


def fit_predict(train_rows, test_rows, threshold: float = 0.5):
    X_tr = [r["text"] for r in train_rows]
    y_tr = [r["label"] for r in train_rows]
    X_te = [r["text"] for r in test_rows]
    y_te = np.array([r["label"] for r in test_rows])

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", max_features=50000)),
        ("model", LogisticRegression(max_iter=2000)),
    ])
    pipe.fit(X_tr, y_tr)

    proba = pipe.predict_proba(X_te)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_te, y_pred, labels=[0, 1]).ravel()
    p, r, f1, _ = precision_recall_fscore_support(y_te, y_pred, average="binary", zero_division=0)

    return {
        "test_n": int(len(y_te)),
        "spam_in_test": int(y_te.sum()),
        "predicted_spam": int(y_pred.sum()),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "precision": float(p), "recall": float(r), "f1": float(f1),
    }


def main():
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data" / "raw"

    rows = load_rows(data_dir)
    rows_sorted = sorted(rows, key=lambda x: x["date"])

    blocks = np.array_split(rows_sorted, 4)  # 3 expanding windows
    train_accum = list(blocks[0])

    print(f"Loaded rows: {len(rows_sorted)}")
    for i in range(1, 4):
        test_block = list(blocks[i])
        metrics = fit_predict(train_accum, test_block, threshold=0.5)
        print(f"\nWindow {i}:")
        print(f"  test_n={metrics['test_n']} spam_in_test={metrics['spam_in_test']} predicted_spam={metrics['predicted_spam']}")
        print(f"  TN={metrics['TN']} FP={metrics['FP']} FN={metrics['FN']} TP={metrics['TP']}")
        print(f"  spam Precision={metrics['precision']:.3f} Recall={metrics['recall']:.3f} F1={metrics['f1']:.3f}")

        train_accum += test_block


if __name__ == "__main__":
    main()
