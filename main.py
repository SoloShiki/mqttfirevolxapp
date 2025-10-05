# main.py
import json
import time
import argparse
import cv2
from infer import EdgeImpulseRunner
from mqtt_client import MqttPublisher

def load_config(path="config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_forever(cfg):
    # Inicializa inferencia
    runner = EdgeImpulseRunner(
        model_path=cfg.get("model_path", "tflite_learn_785526_3.tflite"),
        labels_path=cfg.get("labels_path", "labels.json")
    )

    # Inicializa MQTT
    mqtt = MqttPublisher(
        broker=cfg.get("broker", "localhost"),
        port=cfg.get("broker_port", 1883),
        topic=cfg.get("topic", "edgeimpulse/detections"),
        client_id=cfg.get("client_id", "ei-pc-01")
    )

    # Configuración de detección
    threshold = cfg.get("threshold", 0.3)  # Probabilidad mínima para considerar cigarro
    buffer_size = 5  # número de frames para suavizado
    cigar_buffer = []
    detected = False  # Estado de detección de cigarro para publicación única

    # Captura de cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    print("Iniciando detección en vivo. Presiona 'q' para salir.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Inferencia
            result = runner.infer_frame(frame)
            predictions = result.get("predictions", [])

            # Extraer probabilidad de "cigar"
            prob_cigar = next(
                (p["probability"] for p in predictions if p["label"].lower().strip() == "cigar"), 0.0
            )

            # Suavizado de frames
            cigar_buffer.append(prob_cigar)
            if len(cigar_buffer) > buffer_size:
                cigar_buffer.pop(0)
            avg_prob = sum(cigar_buffer) / len(cigar_buffer)

            # Lógica de publicación
            if avg_prob >= threshold:
                if not detected:
                    message = f"cigar {avg_prob*100:.0f}%"
                    mqtt.publish(message)
                    print("Detección publicada:", message)
                    detected = True
                cv2.putText(frame, f"Cigar {avg_prob*100:.0f}%", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                detected = False  # reinicia detección para el próximo evento

            # Mostrar frame en ventana
            cv2.imshow("Detección en vivo", frame)

            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.01)  # pequeño delay para no saturar CPU

    except KeyboardInterrupt:
        print("Detención manual...")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        mqtt.disconnect()
        runner.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config.json", help="Ruta al archivo de configuración JSON")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_forever(cfg)
