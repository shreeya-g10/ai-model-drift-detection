import json
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def build_dataset():
    spam_examples = [
        "Congratulations! You won a free iPhone. Click now.",
        "Claim your cash prize by sending bank details.",
        "Limited offer! Buy now and get 80 percent off.",
        "You have been selected for a lottery reward.",
        "Earn money quickly from home with no effort.",
        "Win big! Reply YES to claim your bonus.",
        "Exclusive deal waiting. Act immediately.",
        "Free vacation tickets available. Register now.",
        "Urgent: verify account and receive reward.",
        "Get rich fast with this secret method.",
    ]
    ham_examples = [
        "Are we still meeting for lunch tomorrow?",
        "Please review the project report by evening.",
        "Your package will be delivered today.",
        "Can you share the lecture notes?",
        "I will call you after the meeting ends.",
        "Let's schedule a doctor appointment next week.",
        "Thanks for your help with the assignment.",
        "Reminder: team standup starts at 10 AM.",
        "Dinner was great, see you soon.",
        "Please find the invoice attached.",
    ]
    texts = spam_examples * 20 + ham_examples * 20
    labels = ["spam"] * (len(spam_examples) * 20) + ["not_spam"] * (len(ham_examples) * 20)
    return texts, labels


def main():
    root = Path(__file__).resolve().parent
    model_path = root / "model.pkl"
    metrics_path = root / "baseline_metrics.json"

    texts, labels = build_dataset()
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