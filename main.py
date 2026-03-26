import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime

os.makedirs("defects", exist_ok=True)

MAX_ALLOWED_DEFECT_AREA = 500


def detect_defects_and_annotate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    defects = []
    defect_id = 1

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)

            defects.append((x, y, w, h, area))

            # color based on tolerance
            if area > MAX_ALLOWED_DEFECT_AREA:
                color = (0, 0, 255)  # red
                label = f"DEFECT {defect_id} - REJECT"
            else:
                color = (0, 255, 255)  # yellow
                label = f"DEFECT {defect_id}"

            # draw rectangle
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)

            # draw label
            cv2.putText(
                image,
                f"{label} | Area:{int(area)}",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

            defect_id += 1

    return image, defects


st.title("🏭 AI Leather Defect Inspection")

start = st.button("▶ Start Inspection")
stop = st.button("⏹ Stop")

frame_placeholder = st.empty()
status_placeholder = st.empty()
stats_placeholder = st.empty()

total_count = 0
reject_count = 0

if start:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_count += 1

        annotated, defects = detect_defects_and_annotate(frame)

        reject = any(d[4] > MAX_ALLOWED_DEFECT_AREA for d in defects)

        if reject:
            status = "❌ NOT OK"
            reject_count += 1

            filename = datetime.now().strftime("defects/reject_%H%M%S.jpg")
            cv2.imwrite(filename, annotated)

        else:
            status = "✅ OK"

        reject_rate = (reject_count / total_count) * 100

        frame_placeholder.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            channels="RGB"
        )

        status_placeholder.markdown(f"# {status}")

        stats_placeholder.markdown(f"""
### 📊 Inspection Stats
Total: {total_count}  
Rejected: {reject_count}  
Reject Rate: {reject_rate:.2f}%  
Defects: {len(defects)}
""")

        if stop:
            break

    cap.release()
