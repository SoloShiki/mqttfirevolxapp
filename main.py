import argparse
import cv2
import json
import numpy as np
import paho.mqtt.client as mqtt
import time

# Parameters
ALPHA = 0.5
PROB_THRESHOLD = 0.5
MIN_AREA = 200
MAX_AREA = 10000

# Strict dark brown HSV for cigar
HSV_LOWER = np.array([10, 120, 50])
HSV_UPPER = np.array([20, 200, 120])

def get_dark_brown_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config,"r") as f:
        cfg = json.load(f)

    client = mqtt.Client()
    try:
        client.connect(cfg.get("mqtt_host","localhost"), cfg.get("mqtt_port",1883), 60)
        client.loop_start()
        print("MQTT connected")
    except:
        print("MQTT failed")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found")
        return

    smoothed_prob = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        mask = get_dark_brown_mask(frame)
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Pick the largest contour only
        best_cnt = None
        max_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if MIN_AREA <= area <= MAX_AREA and area > max_area:
                max_area = area
                best_cnt = cnt

        smoothed_prob = ALPHA*(1.0 if best_cnt is not None else 0.0) + (1-ALPHA)*smoothed_prob

        # Draw a single bounding box only
        if best_cnt is not None and smoothed_prob > 0:
            rect = cv2.minAreaRect(best_cnt)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.drawContours(frame,[box],0,(0,0,255),2)

            # Optional: highlight inside the bounding box
            overlay = frame.copy()
            mask_roi = np.zeros(frame.shape[:2], np.uint8)
            cv2.drawContours(mask_roi,[best_cnt],-1,255,-1)
            overlay[mask_roi==255] = (0,255,255)
            frame = cv2.addWeighted(frame,0.7,overlay,0.3,0)

        cv2.putText(frame,f"Cigar: {smoothed_prob:.2f}",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow("Cigar Detector - Single Box", frame)

        if smoothed_prob >= PROB_THRESHOLD:
            msg = json.dumps({"label":"cigar","probability":smoothed_prob,"time":time.time()})
            try:
                client.publish(cfg.get("mqtt_topic","cigar/detect"), msg)
            except:
                pass

        if cv2.waitKey(1) & 0xFF==27:
            break

    cap.release()
    cv2.destroyAllWindows()
    client.loop_stop()
    client.disconnect()

if __name__=="__main__":
    main()
