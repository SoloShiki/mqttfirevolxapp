# infer.py
import tensorflow as tf
import numpy as np
import cv2
import json
import time

class EdgeImpulseRunner:
    def __init__(self, model_path="tflite_learn_785526_3.tflite", labels_path="labels.json"):
        # Cargar modelo TFLite
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Obtener detalles de entrada/salida
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Cargar labels como diccionario {index: label_name}
        self.labels = {}
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                labels_data = json.load(f)
            for item in labels_data:
                idx = int(item["index"])
                label_name = item["label"]
                self.labels[idx] = label_name
        except Exception as e:
            print(f"No se pudo cargar labels.json: {e}")

    def preprocess(self, frame):
        """
        Redimensiona la imagen a 96x96, normaliza y agrega dimensión batch.
        """
        target_h, target_w = 96, 96
        # Opcional: recortar la región central para que el cigarro ocupe más píxeles
        h_cam, w_cam, _ = frame.shape
        x1, y1 = w_cam//4, h_cam//4
        x2, y2 = 3*w_cam//4, 3*h_cam//4
        frame_cropped = frame[y1:y2, x1:x2]

        img = cv2.resize(frame_cropped, (target_w, target_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)

    def infer_frame(self, frame):
        """
        Realiza inferencia sobre un frame de OpenCV y devuelve predicciones.
        """
        input_data = self.preprocess(frame)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        output_flat = np.array(output_data).flatten()

        predictions = []
        for i, prob in enumerate(output_flat):
            prob_value = float(prob)
            predictions.append({
                "label": self.labels.get(i, f"class_{i}"),
                "probability": prob_value
            })

        return {"timestamp": time.time(), "predictions": predictions}

    def close(self):
        del self.interpreter
