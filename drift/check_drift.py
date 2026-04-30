import argparse
import sqlite3
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "backend" / "predictions.db"


def fetch_predictions(conn, offset: int, limit: int):
    rows = conn.execute(
        """
        SELECT prediction
        FROM prediction_logs
        ORDER BY id ASC
        LIMIT ? OFFSET ?
        """,
        (limit, offset),
    ).fetchall()
    return [row[0] for row in rows]


def distribution(values):
    counts = Counter(values)
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def main():
    parser = argparse.ArgumentParser(description="Simple prediction-distribution drift checker")
    parser.add_argument("--window-size", type=int, default=100, help="Records per comparison window")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="Absolute change threshold for any class proportion",
    )
    args = parser.parse_args()

    if not DB_PATH.exists():
        print("No SQLite DB found yet. Run predictions first.")
        return

    with sqlite3.connect(DB_PATH) as conn:
        total = conn.execute("SELECT COUNT(*) FROM prediction_logs").fetchone()[0]
        needed = args.window_size * 2
        if total < needed:
            print(
                "Not enough data for drift detection yet. "
                f"Have {total} prediction log row(s); need at least {needed} "
                f"(two windows of --window-size {args.window_size}). "
                "Send POST /predict requests while the API is running."
            )
            return

        baseline = fetch_predictions(conn, total - (args.window_size * 2), args.window_size)
        current = fetch_predictions(conn, total - args.window_size, args.window_size)

    base_dist = distribution(baseline)
    curr_dist = distribution(current)
    all_labels = sorted(set(base_dist.keys()) | set(curr_dist.keys()))

    max_shift = 0.0
    for label in all_labels:
        shift = abs(curr_dist.get(label, 0.0) - base_dist.get(label, 0.0))
        max_shift = max(max_shift, shift)
        print(
            f"{label}: baseline={base_dist.get(label, 0.0):.3f}, "
            f"current={curr_dist.get(label, 0.0):.3f}, shift={shift:.3f}"
        )

    if max_shift >= args.threshold:
        print("Drift detected!")
    else:
        print("No drift detected.")


if __name__ == "__main__":
    main()
