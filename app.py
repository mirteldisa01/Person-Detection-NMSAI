from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO
import cv2
import os
import urllib.request
import base64

# ================= CONFIG =================
CONF_THRESHOLD = 0.5
INTERVAL_SEC = 0.5   # ambil frame tiap 0.5 detik (lebih realistis dari 0.01)
MAX_SHOWN = 3

MODEL_PATH = "person-x-150.pt"
MODEL_URL = "https://github.com/mirteldisa01/Person-Detection-NMSAI/releases/download/v1/person-x-150.pt"

app = FastAPI(title="Person Detection API")

# ================= LOAD MODEL =================
def load_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return YOLO(MODEL_PATH)

model = load_model()

# ================= REQUEST SCHEMA =================
class VideoRequest(BaseModel):
    video_url: str

# ================= CORE PROCESS =================
def process_video(video_url):
    cap = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        return False, 0.0, []

    best_frames = {}
    person_detected = False
    last_bucket = -1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        bucket = int(ms // (INTERVAL_SEC * 1000))

        if bucket == last_bucket:
            continue
        last_bucket = bucket

        results = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]

        max_conf = 0.0
        found_person = False

        if results.boxes is not None:
            for box in results.boxes:
                cls = int(box.cls)
                conf = float(box.conf)

                if model.names[cls].lower() != "person":
                    continue
                if conf < CONF_THRESHOLD:
                    continue

                found_person = True
                person_detected = True
                max_conf = max(max_conf, conf)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Person {conf:.2f}",
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        if found_person:
            if (
                bucket not in best_frames
                or max_conf > best_frames[bucket]["conf"]
            ):
                best_frames[bucket] = {
                    "conf": max_conf,
                    "frame": frame.copy(),
                }

    cap.release()

    # ===== Ambil TOP N terbaik =====
    frames_sorted = list(best_frames.items())
    frames_sorted.sort(key=lambda x: x[1]["conf"], reverse=True)
    frames_sorted = frames_sorted[:MAX_SHOWN]

    image_list = []
    max_conf_global = 0.0

    for _, data in frames_sorted:
        max_conf_global = max(max_conf_global, data["conf"])
        _, buffer = cv2.imencode(".jpg", data["frame"])
        image_base64 = base64.b64encode(buffer).decode("utf-8")
        image_list.append(image_base64)

    return person_detected, max_conf_global, image_list


# ================= API ENDPOINT =================
@app.post("/detect")
def detect_person(data: VideoRequest):
    detected, max_conf, images = process_video(data.video_url)

    return {
        "status": "PERSON DETECTED" if detected else "CLEAR",
        "person_detected": detected,
        "max_confidence": round(max_conf, 4),
        "total_images": len(images),
        "images_base64": images
    }
