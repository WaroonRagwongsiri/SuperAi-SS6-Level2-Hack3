import math
import os
import pickle
from collections import deque

import numpy as np


DEFAULT_WINDOW_SIZE = 50
DEFAULT_THRESHOLD = 0.5


def _stats(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return [0.0, 0.0, 0.0, 0.0]

    return [
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.max(arr)),
        float(np.min(arr)),
    ]


def extract_window_features(samples):
    ax = [row["ax"] for row in samples]
    ay = [row["ay"] for row in samples]
    az = [row["az"] for row in samples]
    gx = [row["gx"] for row in samples]
    gy = [row["gy"] for row in samples]
    gz = [row["gz"] for row in samples]
    peak = [row["peak"] for row in samples]

    acc_mag = [
        math.sqrt(row["ax"] ** 2 + row["ay"] ** 2 + row["az"] ** 2)
        for row in samples
    ]
    gyro_mag = [
        math.sqrt(row["gx"] ** 2 + row["gy"] ** 2 + row["gz"] ** 2)
        for row in samples
    ]

    features = []
    for values in (ax, ay, az, gx, gy, gz, acc_mag, gyro_mag, peak):
        features.extend(_stats(values))

    if len(samples) >= 2:
        duration_ms = max(1.0, samples[-1]["time_ms"] - samples[0]["time_ms"])
        features.append(float(len(samples) / duration_ms * 1000.0))
    else:
        features.append(0.0)

    return np.asarray(features, dtype=float)


class SklearnPunchPredictor:
    def __init__(self, model_path, window_size=DEFAULT_WINDOW_SIZE, threshold=DEFAULT_THRESHOLD):
        self.model_path = model_path
        self.window_size = int(window_size)
        self.threshold = float(threshold)
        self.samples = deque(maxlen=self.window_size)
        self.model = None
        self.ready = False
        self.error = ""

        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            self.error = f"Model file not found: {self.model_path}"
            return

        try:
            with open(self.model_path, "rb") as f:
                payload = pickle.load(f)

            if isinstance(payload, dict):
                self.model = payload.get("model")
                self.threshold = float(payload.get("threshold", self.threshold))
                self.window_size = int(payload.get("window_size", self.window_size))
                self.samples = deque(self.samples, maxlen=self.window_size)
            else:
                self.model = payload

            if self.model is None:
                raise ValueError("Model payload does not contain a model")

            self.ready = True
            self.error = ""
        except Exception as e:
            self.ready = False
            self.error = str(e)

    def add_sample(self, time_ms, ax, ay, az, gx, gy, gz, peak):
        self.samples.append({
            "time_ms": float(time_ms),
            "ax": float(ax),
            "ay": float(ay),
            "az": float(az),
            "gx": float(gx),
            "gy": float(gy),
            "gz": float(gz),
            "peak": float(peak),
        })

        if len(self.samples) < self.window_size:
            return None

        features = extract_window_features(list(self.samples)).reshape(1, -1)

        if self.ready:
            try:
                if hasattr(self.model, "predict_proba"):
                    return float(self.model.predict_proba(features)[0, 1])

                if hasattr(self.model, "decision_function"):
                    score = float(self.model.decision_function(features)[0])
                    return 1.0 / (1.0 + math.exp(-score))

                return float(self.model.predict(features)[0])
            except Exception as e:
                self.ready = False
                self.error = str(e)

        return self._heuristic_probability(features[0])

    def _heuristic_probability(self, features):
        acc_mag_max = features[26]
        gyro_mag_max = features[30]
        peak_max = features[34]

        acc_score = min(max((acc_mag_max - 1.35) / 1.6, 0.0), 1.0)
        gyro_score = min(max((gyro_mag_max - 90.0) / 360.0, 0.0), 1.0)
        peak_score = min(max((peak_max - 0.35) / 1.2, 0.0), 1.0)

        return float((acc_score * 0.35) + (gyro_score * 0.45) + (peak_score * 0.20))
