import math
import json
import os
import pickle
from collections import deque

import numpy as np

try:
    import torch
except Exception:
    torch = None


DEFAULT_WINDOW_SIZE = 50
DEFAULT_THRESHOLD = 0.5
MAX_SCORE = 999.0


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


class TorchScriptPunchPredictor:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.metadata_path = os.path.join(model_dir, "metadata.json")
        self.model_path = os.path.join(model_dir, "model_scripted.pt")
        self.ready = False
        self.error = ""
        self.model = None
        self.features = []
        self.window_size = 128
        self.step = 64
        self.threshold = DEFAULT_THRESHOLD
        self.mean = None
        self.std = None

        self._load_model()

    def _load_model(self):
        if torch is None:
            self.error = "PyTorch is required to load the punch LSTM model"
            return

        if not os.path.exists(self.metadata_path):
            self.error = f"Metadata file not found: {self.metadata_path}"
            return

        if not os.path.exists(self.model_path):
            self.error = f"Model file not found: {self.model_path}"
            return

        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            self.features = list(metadata["features"])
            self.window_size = int(metadata["window_size"])
            self.step = int(metadata.get("step", max(1, self.window_size // 2)))
            self.threshold = float(metadata.get("threshold", DEFAULT_THRESHOLD))
            self.mean = np.asarray(metadata["mean"], dtype=np.float32)
            self.std = np.asarray(metadata["std"], dtype=np.float32)
            self.model = torch.jit.load(self.model_path, map_location="cpu")
            self.model.eval()
            self.ready = True
            self.error = ""
        except Exception as e:
            self.ready = False
            self.error = str(e)

    def _sample_features(self, samples):
        rows = []
        for row in samples:
            ax = float(row["ax_g"])
            ay = float(row["ay_g"])
            az = float(row["az_g"])
            gx = float(row["gx_dps"])
            gy = float(row["gy_dps"])
            gz = float(row["gz_dps"])
            values = {
                "ax_g": ax,
                "ay_g": ay,
                "az_g": az,
                "gx_dps": gx,
                "gy_dps": gy,
                "gz_dps": gz,
                "acc_mag": math.sqrt((ax * ax) + (ay * ay) + (az * az)),
            }
            rows.append([values.get(feature, 0.0) for feature in self.features])

        return np.asarray(rows, dtype=np.float32)

    def _windows(self, features):
        if len(features) == 0:
            return []

        if len(features) < self.window_size:
            pad_count = self.window_size - len(features)
            pad = np.repeat(features[:1], pad_count, axis=0)
            return [np.vstack([pad, features])]

        windows = []
        start = 0
        while start + self.window_size <= len(features):
            windows.append(features[start:start + self.window_size])
            start += max(1, self.step)

        if windows and len(features) > self.window_size and not np.array_equal(windows[-1], features[-self.window_size:]):
            windows.append(features[-self.window_size:])

        return windows

    def predict_samples(self, samples):
        if not self.ready:
            return {
                "is_punch": False,
                "probability": 0.0,
                "threshold": self.threshold,
                "error": self.error,
            }

        features = self._sample_features(samples)
        windows = self._windows(features)
        if not windows:
            return {"is_punch": False, "probability": 0.0, "threshold": self.threshold}

        batch = np.stack(windows).astype(np.float32)
        normalized = (batch - self.mean) / (self.std + 1e-8)

        with torch.no_grad():
            tensor = torch.from_numpy(normalized)
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        probability = float(np.max(probs))
        return {
            "is_punch": probability >= self.threshold,
            "probability": probability,
            "threshold": self.threshold,
        }

    def score_samples(self, samples):
        prediction = self.predict_samples(samples)
        if not prediction.get("is_punch"):
            return {
                "overall": 0.0,
                "metrics": prediction,
                "scores": {"punch_probability": round(prediction.get("probability", 0.0) * MAX_SCORE, 1)},
                "max_score": MAX_SCORE,
                "error": prediction.get("error", "model predicted rest"),
            }

        peak_values = np.asarray([float(row.get("peak", 0.0)) for row in samples], dtype=float)
        acc_values = np.asarray([
            math.sqrt(
                (float(row["ax_g"]) ** 2) +
                (float(row["ay_g"]) ** 2) +
                (float(row["az_g"]) ** 2)
            )
            for row in samples
        ], dtype=float)
        gyro_values = np.asarray([
            math.sqrt(
                (float(row["gx_dps"]) ** 2) +
                (float(row["gy_dps"]) ** 2) +
                (float(row["gz_dps"]) ** 2)
            )
            for row in samples
        ], dtype=float)

        peak_signal = float(peak_values.max()) if peak_values.size else 0.0
        acc_peak_g = float(acc_values.max()) if acc_values.size else 0.0
        gyro_peak_dps = float(gyro_values.max()) if gyro_values.size else 0.0
        punch_probability = float(prediction["probability"])

        force_score = np.clip((peak_signal - 0.15) / 1.35, 0.0, 1.0) * MAX_SCORE
        motion_score = np.clip((acc_peak_g - 1.0) / 4.0, 0.0, 1.0) * MAX_SCORE
        confidence_score = np.clip(punch_probability, 0.0, 1.0) * MAX_SCORE
        overall = (force_score * 0.65) + (motion_score * 0.20) + (confidence_score * 0.15)

        metrics = {
            "punch_probability": punch_probability,
            "peak_signal": peak_signal,
            "acc_peak_g": acc_peak_g,
            "gyro_peak_dps": gyro_peak_dps,
        }
        scores = {
            "force": float(round(force_score, 1)),
            "motion": float(round(motion_score, 1)),
            "confidence": float(round(confidence_score, 1)),
        }

        return {
            "overall": float(round(np.clip(overall, 0.0, MAX_SCORE), 1)),
            "metrics": metrics,
            "scores": scores,
            "max_score": MAX_SCORE,
        }
