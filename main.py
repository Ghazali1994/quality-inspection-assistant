import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_paste_button import paste_image_button
from ultralytics import YOLO

# -------------------------------
# Load YOLO model (load once)
# -------------------------------
# Replace with your trained model later:
# model = YOLO("runs/detect/train20/weights/best.pt")
model = YOLO("yolov8n.pt")


# -------------------------------
# YOLO Defect Detection Function
# -------------------------------
def detect_defects_and_annotate(image):
    results = model(image)

    defects = []

    for r in results:
        boxes = r.boxes

        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            w = x2 - x1
            h = y2 - y1

            defects.append((x1, y1, w, h, conf))

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Label
            label = f"Defect {conf:.2f}"
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

    return image, defects


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="AI Leather Defect Detection Tool")
st.title("AI Leather Defect Detection Tool")
st.write("Upload, Capture, or Paste an image to inspect for defects.")

# selector
option = st.radio(
    "Choose input method:",
    ["Upload Image", "Capture from Camera", "Paste Image"]
)

image = None

# -------------------------------
# Upload
# -------------------------------
if option == "Upload Image":
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# -------------------------------
# Camera
# -------------------------------
elif option == "Capture from Camera":
    camera_image = st.camera_input("Capture Image")

    if camera_image is not None:
        bytes_data = camera_image.getvalue()
        file_bytes = np.asarray(
            bytearray(bytes_data),
            dtype=np.uint8
        )
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# -------------------------------
# Paste
# -------------------------------
elif option == "Paste Image":
    pasted = paste_image_button("📋 Paste Image")

    if pasted.image_data is not None:
        pil_image = pasted.image_data
        image = cv2.cvtColor(
            np.array(pil_image),
            cv2.COLOR_RGB2BGR
        )

# -------------------------------
# Run detection
# -------------------------------
if image is not None:

    annotated_image, defects = detect_defects_and_annotate(image.copy())

    st.image(
        cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
        caption="Annotated Image",
        use_column_width=True
    )

    st.markdown(f"### 🧪 {len(defects)} defect(s) found")

    for idx, (x, y, w, h, conf) in enumerate(defects, 1):
        st.write(
            f"**Defect {idx}:** "
            f"Location: (x={x}, y={y}) | "
            f"Size: {w}x{h} | "
            f"Confidence: {conf:.2f}"
        )
