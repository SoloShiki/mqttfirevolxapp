import cv2
import json
import argparse
import time
import numpy as np

from mqtt_client import MqttPublisher

# --- PARÁMETROS DE VISUALIZACIÓN ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
BBOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 255, 0)
INFO_COLOR = (255, 255, 0)

# --- PARÁMETROS DE DETECCIÓN ---
MATCH_THRESHOLD = 0.79
TRACKER_TYPE = "CSRT"
BBOX_CROP_OFFSETS = (0.1, 0.2, 0.1, 0.2)
HSV_CIGAR_LOWER = np.array([5, 120, 50])
HSV_CIGAR_UPPER = np.array([30, 255, 255])
MIN_CIGAR_PIXEL_PERCENT = 0.3
HIGH_CONFIDENCE_ALERT_THRESHOLD = 0.78

def create_tracker():
    tracker_map = {
        "CSRT": cv2.TrackerCSRT_create,
        "KCF": cv2.TrackerKCF_create,
        "MIL": cv2.TrackerMIL_create
    }
    return tracker_map.get(TRACKER_TYPE, cv2.TrackerCSRT_create)()

def main():
    parser = argparse.ArgumentParser(description="Robust Cigar Detection and Tracking")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the config JSON file.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    try:
        template = cv2.imread('cigar_template.jpg', 0)
        if template is None:
            print("[ERROR] No se pudo cargar la plantilla 'cigar_template.jpg'.")
            return
        tw, th = template.shape[::-1]
        print(f"[INFO] Plantilla cargada. Dimensiones: {tw}x{th}")
    except Exception as e:
        print(f"[ERROR] Error al cargar la plantilla: {e}")
        return

    print("[INFO] Connecting to MQTT broker...")
    mqtt_client = MqttPublisher(
        broker=cfg['mqtt_host'],
        port=cfg['mqtt_port'],
        topic=cfg['mqtt_topic']
    )
    if not mqtt_client.connected:
        print("[ERROR] Could not connect to MQTT broker. Exiting.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera not found.")
        return

    print("[INFO] Starting camera feed. Press 'ESC' to exit.")

    last_publish_time = 0
    publish_cooldown = 2
    tracker = None
    bbox = None
    detection_mode = True

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Camera frame could not be read. Exiting loop.")
                break

            current_frame_has_cigar = False
            max_val = 0

            if detection_mode:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)

                if max_val >= MATCH_THRESHOLD:
                    x, y = max_loc
                    w, h = tw, th
                    roi = frame[y:y+h, x:x+w]
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    mask_color = cv2.inRange(hsv_roi, HSV_CIGAR_LOWER, HSV_CIGAR_UPPER)
                    cigar_pixels = cv2.countNonZero(mask_color)
                    total_pixels = w * h
                    
                    color_check_passed = total_pixels > 0 and (cigar_pixels / total_pixels) >= MIN_CIGAR_PIXEL_PERCENT
                    
                    if color_check_passed:
                        bbox = (x, y, w, h)
                        tracker = create_tracker()
                        tracker.init(frame, bbox)
                        detection_mode = False
                        current_frame_has_cigar = True
                        print(f"[INFO] Detección VÁLIDA por TM ({max_val*100:.0f}%) y color.")
                    else:
                        cv2.putText(frame, f"TM hit ({max_val*100:.0f}%) - Color Rejected", (10, 60), FONT, 0.7, INFO_COLOR, 2)
                        
                        if max_val >= HIGH_CONFIDENCE_ALERT_THRESHOLD:
                            current_time = time.time()
                            if (current_time - last_publish_time) > publish_cooldown:
                                
                                # --- INICIO DE LA CORRECCIÓN DE TEXTO ---
                                # El payload ahora es genérico y no menciona el fallo de color.
                                payload = json.dumps({
                                    "label": "cigar", "status": "detected",
                                    "confidence": round(max_val, 2),
                                    "timestamp": current_time
                                })
                                print(f"[MQTT] Publicando por alta confianza de TM ({max_val*100:.0f}%).")
                                # --- FIN DE LA CORRECCIÓN DE TEXTO ---
                                
                                mqtt_client.publish(payload)
                                last_publish_time = current_time
                
                if not current_frame_has_cigar:
                    cv2.putText(frame, "Searching...", (10, 30), FONT, 0.7, INFO_COLOR, 2)

            else: # Modo seguimiento
                if tracker is not None:
                    success, new_bbox = tracker.update(frame)
                    if success:
                        bbox = new_bbox
                        current_frame_has_cigar = True
                        cv2.putText(frame, "Tracking...", (10, 30), FONT, 0.7, INFO_COLOR, 2)
                    else:
                        print("[WARN] Tracker perdió el cigarro. Volviendo a buscar.")
                        detection_mode = True
                        tracker = None
                        bbox = None
                else:
                    detection_mode = True

            if current_frame_has_cigar and bbox is not None:
                x, y, w, h = [int(v) for v in bbox]
                
                x_start_crop = int(w * BBOX_CROP_OFFSETS[0])
                y_start_crop = int(h * BBOX_CROP_OFFSETS[1])
                x_end_crop = int(w * BBOX_CROP_OFFSETS[2])
                y_end_crop = int(h * BBOX_CROP_OFFSETS[3])

                draw_x1 = x + x_start_crop
                draw_y1 = y + y_start_crop
                draw_x2 = x + w - x_end_crop
                draw_y2 = y + h - y_end_crop

                cv2.rectangle(frame, (draw_x1, draw_y1), (draw_x2, draw_y2), BBOX_COLOR, 2)
                cv2.putText(frame, "Cigar Detected", (draw_x1, draw_y1 - 10), FONT, 0.7, TEXT_COLOR, 2)

                current_time = time.time()
                if (current_time - last_publish_time) > publish_cooldown:
                    payload = json.dumps({
                        "label": "cigar", "status": "detected_and_tracking",
                        "bbox": [draw_x1, draw_y1, draw_x2, draw_y2],
                        "confidence": round(max_val if detection_mode else 1.0, 2),
                        "timestamp": current_time
                    })
                    print(f"[MQTT] Publishing: {payload}")
                    mqtt_client.publish(payload)
                    last_publish_time = current_time
            
            cv2.imshow("Cigar Detector", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        print("[INFO] Shutting down...")
        cap.release()
        cv2.destroyAllWindows()
        mqtt_client.disconnect()

if __name__ == "__main__":
    main()