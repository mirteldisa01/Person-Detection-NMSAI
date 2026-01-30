import streamlit as st
import cv2
import os
import urllib.request
from ultralytics import YOLO

# ================= CONFIG =================

st.set_page_config(page_title="Person Detection", layout="wide")

# ===== BACKEND PARAMETERS =====
CONF_THRESHOLD = 0.5
MAX_SHOWN = 3
INTERVAL_SEC = 0.01

MODEL_PATH = "person-x-150.pt"
MODEL_URL = "https://github.com/mirteldisa01/Person-Detection-NMSAI/releases/download/v1/person-x-150.pt"

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return YOLO(MODEL_PATH)

model = load_model()

# ================= VIDEO PROCESS =================
def process_video(cap):
    best_frame_per_bucket = {}
    person_detected = False
    last_bucket = -1

    # ===== Ambil frame pertama sebagai backup =====
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return {}, None, False

    first_frame_backup = first_frame.copy()

    # Reset video ke awal
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

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
        found_person_in_frame = False

        if results.boxes is not None:
            for box in results.boxes:
                cls = int(box.cls)
                conf = float(box.conf)

                # ==== FILTER PERSON ONLY ====
                if model.names[cls].lower() != "person":
                    continue
                if conf < CONF_THRESHOLD:
                    continue

                found_person_in_frame = True
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

        # ===== SIMPAN FRAME HANYA JIKA ADA PERSON =====
        if found_person_in_frame:
            if (
                bucket not in best_frame_per_bucket
                or max_conf > best_frame_per_bucket[bucket]["conf"]
            ):
                best_frame_per_bucket[bucket] = {
                    "conf": max_conf,
                    "frame": frame.copy(),
                }

    cap.release()
    return best_frame_per_bucket, first_frame_backup, person_detected

# ================= UI =================
st.title("🚨 Person Detection")

video_url = st.text_input(
    "Masukkan URL Video (.mp4 / .webm)",
    placeholder="https://example.com/video.webm"
)

if video_url and st.button("Process Video"):
    with st.spinner("Processing video..."):
        cap = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            st.error("The video cannot be opened. Please make sure the URL is valid and publicly accessible.")
        else:
            best_frames, first_frame, person_detected = process_video(cap)

            status = "PERSON DETECTED" if person_detected else "CLEAR"
            st.subheader(f"Final Status: {status}")

            if person_detected and best_frames:
                frames = list(best_frames.items())
                frames.sort(key=lambda x: x[1]["conf"], reverse=True)
                frames = frames[:MAX_SHOWN]
                frames.sort(key=lambda x: x[0])

                cols = st.columns(len(frames))
                for col, (_, data) in zip(cols, frames):
                    frame = cv2.cvtColor(data["frame"], cv2.COLOR_BGR2RGB)
                    col.image(frame, use_container_width=True)

            else:
                st.info("Final Status: No person detected")

                if first_frame is not None:
                    frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

                    left, center, right = st.columns([30, 40, 30])

                    with center:
                        st.image(
                            frame,
                            #caption="First frame (backup)",
                            use_container_width=True
                        )
