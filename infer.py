import cv2
import numpy as np
import tensorflow as tf
import time
import json
from typing import Dict, Any


class TFLiteModel:
    def __init__(self, model_path: str, labels_path: str = None):
        self.model_path = model_path
        self.labels = self._load_labels(labels_path) if labels_path else None

        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_index = self.input_details[0]['index']
        self.output_index = self.output_details[0]['index']

        # Input info
        self.input_shape = self.input_details[0]['shape']
        self.input_dtype = self.input_details[0]['dtype']

        print(f"[INFO] Input shape: {self.input_shape}, dtype: {self.input_dtype}")
        print(f"[INFO] Output shape: {self.interpreter.get_tensor(self.output_index).shape}")

    def _load_labels(self, labels_path: str):
        try:
            with open(labels_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            return None

        if isinstance(data, list):
            labels = {}
            for x in data:
                if isinstance(x, dict) and "index" in x and "label" in x:
                    labels[int(x["index"])] = str(x["label"])  # keep only string
                elif isinstance(x, str):
                    labels[len(labels)] = x
            return labels
        elif isinstance(data, dict):
            return {int(k): str(v) for k, v in data.items()}
        return None

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        h, w = self.input_shape[1], self.input_shape[2]
        resized = cv2.resize(frame, (w, h))
        img = resized.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)

    def infer_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        ts = time.time()
        try:
            input_data = self.preprocess(frame)
            self.interpreter.set_tensor(self.input_index, input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_index)
        except Exception as e:
            return {"timestamp": ts, "error": str(e), "predictions": []}

        # Example: output shape (1,12,12,2)
        out = np.array(output_data)
        out = out.reshape(-1, out.shape[-1])  # flatten grid → (144,2)

        # Average across cells
        probs = out.mean(axis=0)

        # Normalize
        if not np.isclose(probs.sum(), 1.0, atol=0.1):
            exps = np.exp(probs - np.max(probs))
            probs = exps / exps.sum()

        preds = []
        for idx, p in enumerate(probs):
            label = self.labels.get(idx) if self.labels else str(idx)
            preds.append({
                "index": idx,
                "label": label,
                "probability": float(p)
            })

        preds = sorted(preds, key=lambda x: x["probability"], reverse=True)
        return {"timestamp": ts, "predictions": preds}
