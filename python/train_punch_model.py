import argparse
import os
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from punch_model import DEFAULT_THRESHOLD, DEFAULT_WINDOW_SIZE, extract_window_features


COLUMN_ALIASES = {
    "time_ms": ["time_ms", "phase_elapsed_ms", "timestamp_ms"],
    "ax": ["ax", "ax_g"],
    "ay": ["ay", "ay_g"],
    "az": ["az", "az_g"],
    "gx": ["gx", "gx_dps"],
    "gy": ["gy", "gy_dps"],
    "gz": ["gz", "gz_dps"],
    "peak": ["peak"],
}


def _pick_column(df, names, default=None):
    for name in names:
        if name in df.columns:
            return name
    return default


def _label_values(df):
    if "label" in df.columns:
        return df["label"].astype(int).tolist()

    if "state" in df.columns:
        return df["state"].astype(str).str.lower().eq("punching").astype(int).tolist()

    if "phase" in df.columns:
        return df["phase"].astype(str).str.lower().eq("punch").astype(int).tolist()

    raise ValueError("CSV must contain label, state, or phase column")


def load_samples(csv_path):
    df = pd.read_csv(csv_path)

    columns = {
        key: _pick_column(df, aliases)
        for key, aliases in COLUMN_ALIASES.items()
    }

    missing = [key for key, value in columns.items() if value is None and key != "peak"]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    labels = _label_values(df)
    peak_column = columns["peak"]

    samples = []
    for idx, row in df.iterrows():
        samples.append({
            "time_ms": float(row[columns["time_ms"]]),
            "ax": float(row[columns["ax"]]),
            "ay": float(row[columns["ay"]]),
            "az": float(row[columns["az"]]),
            "gx": float(row[columns["gx"]]),
            "gy": float(row[columns["gy"]]),
            "gz": float(row[columns["gz"]]),
            "peak": float(row[peak_column]) if peak_column else 0.0,
            "label": int(labels[idx]),
        })

    return samples


def build_windows(samples, window_size, stride):
    features = []
    labels = []

    for start in range(0, len(samples) - window_size + 1, stride):
        window = samples[start:start + window_size]
        features.append(extract_window_features(window))
        labels.append(1 if any(row["label"] for row in window) else 0)

    return features, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Labeled IMU CSV file")
    parser.add_argument("--output", default="punch_model_sklearn.pkl")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    samples = load_samples(args.csv)
    features, labels = build_windows(samples, args.window_size, args.stride)

    if not features:
        raise ValueError("Not enough rows to build any training window")

    if len(set(labels)) < 2:
        raise ValueError("Training data must contain both punch and non-punch labels")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit(features, labels)

    payload = {
        "model": model,
        "threshold": args.threshold,
        "window_size": args.window_size,
    }

    output_path = os.path.abspath(args.output)
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"Saved sklearn punch model: {output_path}")


if __name__ == "__main__":
    main()
