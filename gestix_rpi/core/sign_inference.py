"""
Inferență ML pentru semne — versiune Raspberry Pi.

Suportă modele .tflite antrenate cu Gestix Trainer.
Folosește tflite-runtime (ușor) cu fallback la tensorflow.

Auto-încarcă lângă model:
  {model}.labels.json        — clasele (mapping output → nume)
  {model}_preprocessor.json  — normalizare (adaptive / imu_only)

LUCIAN: pune calea modelului în config.py (MODEL_STATIC_PATH).
Dacă modelul e None, inferența returnează ("", 0.0) — UI merge fără ML.

Indici senzori (frame de 21, mâna dreaptă):
  0-14  : Hall
  15-20 : IMU (AX,AY,AZ,GX,GY,GZ)
"""

import json
import os
from typing import List, Optional

import numpy as np

IMU_IDX = [15, 16, 17, 18, 19, 20]
HALL_IDX = list(range(15))


def _load_interpreter(path: str):
    """Încarcă un interpreter TFLite (tflite-runtime sau tensorflow)."""
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter(model_path=path)
    except ImportError:
        import tensorflow as tf
        return tf.lite.Interpreter(model_path=path)


class SignInference:
    """Clasificator semne din model .tflite."""

    def __init__(self):
        self._interp = None
        self._in_index = None
        self._out_index = None
        self.window = 1
        self.n_features = 21
        self.classes: List[str] = []
        self.model_path = ""

        # Normalizare
        self._imu_only = False
        self._imu_scale = None         # np.array (6,)
        self._hall_mean = None         # np.array (15,) — adaptive
        self._hall_scale = None
        self._accel_scale = 20000.0
        self._gyro_scale = 30000.0
        self._hall_max = 4095.0

    @property
    def loaded(self) -> bool:
        return self._interp is not None

    # ── Încărcare ────────────────────────────────────────────

    def load(self, model_path: str) -> None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)

        self._interp = _load_interpreter(model_path)
        self._interp.allocate_tensors()
        in_d = self._interp.get_input_details()
        out_d = self._interp.get_output_details()
        self._in_index = in_d[0]["index"]
        self._out_index = out_d[0]["index"]
        shape = in_d[0]["shape"]   # (1, window, n_features)
        self.window = int(shape[1])
        self.n_features = int(shape[2])
        self.model_path = model_path

        base = os.path.splitext(model_path)[0]
        self._load_labels(base + ".labels.json")
        self._load_preprocessor(base + "_preprocessor.json")

    def _load_labels(self, path: str):
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self.classes = list(json.load(f).get("classes", []))
            except Exception:
                self.classes = []
        if not self.classes:
            self.classes = [f"Clasa {i}" for i in range(20)]

    def _load_preprocessor(self, path: str):
        if not os.path.isfile(path):
            return
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            return
        self._imu_only = bool(d.get("imu_only", False))
        if d.get("imu_scale"):
            self._imu_scale = np.array(d["imu_scale"], dtype=np.float32)
        if d.get("hall_mean") and d.get("hall_scale"):
            self._hall_mean = np.array(d["hall_mean"], dtype=np.float32)
            self._hall_scale = np.maximum(np.array(d["hall_scale"], dtype=np.float32), 1.0)
        self._accel_scale = float(d.get("accel_scale", 20000.0))
        self._gyro_scale = float(d.get("gyro_scale", 30000.0))
        self._hall_max = float(d.get("hall_max", 4095.0))

    # ── Normalizare ──────────────────────────────────────────

    def _normalize(self, frames: np.ndarray) -> np.ndarray:
        """frames: (N, 21) RAW → normalizat conform modelului."""
        frames = frames.astype(np.float32)

        if self._imu_only:
            imu = frames[:, IMU_IDX]
            scale = self._imu_scale if self._imu_scale is not None \
                else np.array([20000.0] * 6, dtype=np.float32)
            return np.clip(imu / scale, -3.0, 3.0).astype(np.float32)

        out = frames.copy()
        # Hall
        if self._hall_mean is not None:
            out[:, HALL_IDX] = (out[:, HALL_IDX] - self._hall_mean) / self._hall_scale
            out[:, HALL_IDX] = np.clip(out[:, HALL_IDX], -1.5, 1.5)
        else:
            out[:, HALL_IDX] = out[:, HALL_IDX] / self._hall_max
        # IMU
        out[:, 15:18] = out[:, 15:18] / self._accel_scale
        out[:, 18:21] = out[:, 18:21] / self._gyro_scale
        return out

    # ── Predicție ────────────────────────────────────────────

    def predict(self, frames: np.ndarray) -> tuple:
        """frames: (N, 21) RAW → (class_name, confidence)."""
        if not self.loaded:
            return "", 0.0
        norm = self._normalize(frames)

        # Ajustare la window
        if norm.shape[0] < self.window:
            pad = np.repeat(norm[-1:], self.window - norm.shape[0], axis=0)
            norm = np.vstack([norm, pad])
        elif norm.shape[0] > self.window:
            norm = norm[-self.window:]

        batch = norm[np.newaxis, ...].astype(np.float32)
        self._interp.set_tensor(self._in_index, batch)
        self._interp.invoke()
        probs = self._interp.get_tensor(self._out_index)[0]

        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        name = self.classes[idx] if idx < len(self.classes) else f"Clasa {idx}"
        return name, conf
