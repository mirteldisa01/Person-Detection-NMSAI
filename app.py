from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
import cv2
import os
import urllib.request
import base64
import threading
import time

# ================= CONFIG =================
CONF_THRESHOLD = 0.5 
INTERVAL_SEC = 1.0          # Frame sampling interval (seconds). Increased for CPU efficiency.
MAX_SHOWN = 3               # Maximum number of best frames returned in response
MAX_VIDEO_SECONDS = 10      # Maximum processing duration (seconds)
MAX_FRAMES = 30             # Maximum total frames processed (hard cap to protect CPU)

MODEL_PATH = "person-11m-150.pt"
MODEL_URL = "https://github.com/mirteldisa01/person-detection-nmsai/releases/download/v1.1.0/person-11m-150.pt"

app = FastAPI(title="Person Detection API")

# ================= GLOBAL MODEL =================
model = None
model_lock = threading.Lock()  # Ensures thread-safe inference when multiple requests hit the API

@app.on_event("startup")
def load_model_once():
    """
    This function runs once when the FastAPI application starts.
    It ensures the model file exists locally and loads it into memory.
    """
    global model
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully")

# ================= REQUEST SCHEMA =================
class VideoRequest(BaseModel):
    video_url: str  # Public URL or stream URL of the video source

# ================= CORE PROCESS =================
def process_video(video_url: str):
    """
    Core video processing function.
    - Reads video stream
    - Samples frames based on INTERVAL_SEC
    - Runs YOLO inference
    - Keeps best frames per time bucket
    - Returns top N highest confidence frames (base64 encoded)
    """

    # Open video stream using FFmpeg backend
    cap = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)

    # If video cannot be opened, return empty result safely
    if not cap.isOpened():
        return False, 0.0, []

    best_frames = {}          # Store best frame per time bucket
    person_detected = False   # Global detection flag
    last_bucket = -1          # Prevent duplicate processing within same interval
    frame_count = 0           # Total processed frames counter
    start_time = time.time()  # Processing start time

    try:
        while cap.isOpened():

            # Limit the processing duration (CPU protection)
            if time.time() - start_time > MAX_VIDEO_SECONDS:
                break

            # Limit the total number of frames (hard cap)
            if frame_count >= MAX_FRAMES:
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Get current timestamp in milliseconds
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)

            # Create time bucket based on sampling interval
            # Example: if INTERVAL_SEC = 1.0, process 1 frame per second
            bucket = int(ms // (INTERVAL_SEC * 1000))

            # Skip frame if still in the same time bucket
            if bucket == last_bucket:
                continue
            last_bucket = bucket

            # Thread-safe inference (important for concurrent API requests)
            with model_lock:
                results = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]

            max_conf = 0.0       # Track highest confidence in this frame
            found_person = False # Local flag for this frame

            # If detection boxes exist
            if results.boxes is not None:
                for box in results.boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)

                    # Filter only "person" class
                    if model.names[cls].lower() != "person":
                        continue

                    # Apply confidence threshold filter
                    if conf < CONF_THRESHOLD:
                        continue

                    found_person = True
                    person_detected = True
                    max_conf = max(max_conf, conf)

                    # Draw bounding box for visualization
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw label below bounding box
                    cv2.putText(
                        frame,
                        f"Person {conf:.2f}",
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            # Store only the best frame per bucket
            # If a frame in the same second has higher confidence, replace it
            if found_person:
                if (
                    bucket not in best_frames
                    or max_conf > best_frames[bucket]["conf"]
                ):
                    best_frames[bucket] = {
                        "conf": max_conf,
                        "frame": frame.copy(),  # Copy to prevent memory overwrite
                    }

    finally:
        # Ensure video resource is always released
        cap.release()

    # ===== Take the top N best =====
    # Sort frames by confidence descending and keep only MAX_SHOWN
    frames_sorted = sorted(
        best_frames.items(),
        key=lambda x: x[1]["conf"],
        reverse=True
    )[:MAX_SHOWN]

    image_list = []
    max_conf_global = 0.0

    # Convert selected frames to base64
    for _, data in frames_sorted:
        max_conf_global = max(max_conf_global, data["conf"])

        # Encode frame to JPEG
        _, buffer = cv2.imencode(".jpg", data["frame"])

        # Convert JPEG buffer to base64 string
        image_base64 = base64.b64encode(buffer).decode("utf-8")
        image_list.append(image_base64)

    return person_detected, max_conf_global, image_list


# ================= API ENDPOINT =================
@app.post("/detect")
def detect_person(data: VideoRequest):
    """
    API endpoint:
    Accepts JSON body:
    {
        "video_url": "http://example.com/video.webm"
    }

    Returns detection status + best frames in base64 format.
    """

    # Validate input
    if not data.video_url:
        raise HTTPException(status_code=400, detail="Video URL required")

    try:
        detected, max_conf, images = process_video(data.video_url)

        return {
            "status": "PERSON DETECTED" if detected else "CLEAR",
            "person_detected": detected,
            "max_confidence": round(max_conf, 4),
            "total_images": len(images),
            "images_base64": images
        }

    except Exception as e:
        # Catch unexpected errors and return HTTP 500
        raise HTTPException(status_code=500, detail=str(e))