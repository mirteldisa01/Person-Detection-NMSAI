import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO

import os
import urllib.request
from ultralytics import YOLO

# ================= CONFIG =================
st.set_page_config(page_title="Person Detection", layout="wide")

MODEL_PATH = "person-x-150.pt"
MODEL_URL = "https://github.com/mirteldisa01/Person-Detection-NMSAI/releases/download/v1/person-x-150.pt"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return YOLO(MODEL_PATH)





model = load_model()

# ================= PROSES VIDEO =================
def process_video(cap, model, conf_threshold, interval_sec):
    best_frame_per_bucket = {}
    first_frame_backup = None
    centroid_history = []
    bucket_with_person = set()
    person_detected = False
    last_bucket = -1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        bucket = int(ms // (interval_sec * 1000))

        if first_frame_backup is None:
            first_frame_backup = frame.copy()

        if bucket == last_bucket:
            continue
        last_bucket = bucket

        results = model(frame, conf=conf_threshold, verbose=False)[0]
        max_conf = 0.0
        current_centroids = []

        if results.boxes is not None:
            for box in results.boxes:
                cls = int(box.cls)
                conf = float(box.conf)

                if model.names[cls].lower() != "person":
                    continue

                person_detected = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                current_centroids.append((cx, cy))
                centroid_history.append((cx, cy))
                bucket_with_person.add(bucket)
                max_conf = max(max_conf, conf)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"Person {conf:.2f}",
                            (x1, y2+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        if current_centroids:
            if bucket not in best_frame_per_bucket or max_conf > best_frame_per_bucket[bucket]["conf"]:
                best_frame_per_bucket[bucket] = {
                    "conf": max_conf,
                    "frame": frame.copy()
                }

    cap.release()
    return best_frame_per_bucket, first_frame_backup, person_detected


# ================= UI =================
st.title("🚨 Person Detection Video App")

uploaded_file = st.file_uploader(
    "Upload video (.mp4 / .webm)",
    type=["mp4", "webm"]
)

conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)
interval_sec = st.number_input("Interval (sec)", 0.01, 1.0, 0.01)
max_shown = st.slider("Max Frame Ditampilkan", 1, 5, 3)

# ================= RUN =================
if uploaded_file and st.button("Process Video"):
    with st.spinner("Processing video..."):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)

        best_frames, first_frame, person_detected = process_video(
            cap, model, conf_threshold, interval_sec
        )

    status = "PERSON DETECTED" if person_detected else "CLEAR"
    st.subheader(f"Final Status: {status}")

    frames = list(best_frames.items())
    frames.sort(key=lambda x: x[1]["conf"], reverse=True)
    frames = frames[:max_shown]
    frames.sort(key=lambda x: x[0])

    cols = st.columns(len(frames))
    for col, (_, data) in zip(cols, frames):
        frame = cv2.cvtColor(data["frame"], cv2.COLOR_BGR2RGB)
        col.image(frame, use_container_width=True)
