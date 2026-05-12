import json
import pickle
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

SMS_SPAM_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
)
SMS_COLLECTION_FILENAME = "SMSSpamCollection"


def ensure_sms_spam_file(data_dir: Path) -> Path:
    """Ensure SMS Spam Collection file exists under model/data/; download UCI zip if missing."""
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / f"{SMS_COLLECTION_FILENAME}.txt"
    if target.exists():
        return target

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with urlopen(SMS_SPAM_ZIP_URL) as resp:
            tmp_path.write_bytes(resp.read())
        with zipfile.ZipFile(tmp_path, "r") as zf:
            inner = next(
                (n for n in zf.namelist() if Path(n).name == SMS_COLLECTION_FILENAME),
                None,
            )
            if inner is None:
                raise FileNotFoundError(
                    "SMS spam zip missing SMSSpamCollection; archive layout may have changed."
                )
            target.write_bytes(zf.read(inner))
    finally:
        tmp_path.unlink(missing_ok=True)

    return target


def load_sms_spam_dataset(root: Path) -> tuple[list[str], list[str]]:
    path = ensure_sms_spam_file(root / "data")
    label_map = {"ham": "not_spam", "spam": "spam"}
    texts: list[str] = []
    labels: list[str] = []

    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if "\t" not in line:
                continue
            raw_label, text = line.split("\t", 1)
            raw_label = raw_label.strip().lower()
            text = text.strip()
            if not text or raw_label not in label_map:
                continue
            texts.append(text)
            labels.append(label_map[raw_label])

    return texts, labels


def main():
    root = Path(__file__).resolve().parent
    model_path = root / "model.pkl"
    metrics_path = root / "baseline_metrics.json"

    texts, labels = load_sms_spam_dataset(root)
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "f1_macro": round(float(f1_score(y_test, predictions, average="macro")), 4),
        "task": "spam_detection",
    }

    with model_path.open("wb") as f:
        pickle.dump(pipeline, f)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved baseline metrics to {metrics_path}")


if __name__ == "__main__":
    main()
