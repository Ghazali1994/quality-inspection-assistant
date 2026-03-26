import streamlit as st
import cv2
import numpy as np

def detect_defects_and_annotate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    defects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            defects.append((x, y, w, h))
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return image, defects


st.set_page_config(page_title="🛡️ Live Quality Inspection")
st.title("🛡️ Live Quality Inspection - Real Time")

start = st.button("▶️ Start Camera")
stop = st.button("⏹️ Stop Camera")

frame_placeholder = st.empty()
info_placeholder = st.empty()

if start:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not accessible")
            break

        annotated, defects = detect_defects_and_annotate(frame)

        frame_placeholder.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            channels="RGB"
        )

        info_placeholder.markdown(f"### 🧪 Defects Detected: {len(defects)}")

        # stop button
        if stop:
            break

    cap.release()
