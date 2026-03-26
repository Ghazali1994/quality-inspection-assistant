import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime

# create folder to save defects
os.makedirs("defects", exist_ok=True)

def detect_defects_and_annotate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    defects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200:
            x, y, w, h = cv2.boundingRect(cnt)
            defects.append((x, y, w, h))
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return image, defects


st.title("🏭 AI Quality Inspection System")

start = st.button("▶ Start Inspection")
stop = st.button("⏹ Stop")

frame_placeholder = st.empty()
status_placeholder = st.empty()

if start:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated, defects = detect_defects_and_annotate(frame)

        # decision logic
        if len(defects) == 0:
            status = "✅ OK"
        else:
            status = "❌ NOT OK"

            # save defective image
            filename = datetime.now().strftime("defects/defect_%H%M%S.jpg")
            cv2.imwrite(filename, annotated)

        frame_placeholder.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            channels="RGB"
        )

        status_placeholder.markdown(f"# {status}")
        status_placeholder.markdown(f"### Defects: {len(defects)}")

        if stop:
            break

    cap.release()
