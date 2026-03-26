import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_paste_button import paste_image_button

# -------------------------------
# Improved Defect Detection Function
# -------------------------------
def detect_defects_and_annotate(image):
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Estimate background (smooth leather texture)
    background = cv2.GaussianBlur(gray, (51, 51), 0)
    diff = cv2.absdiff(gray, background)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    # Adaptive threshold based on image statistics
    mean, std = cv2.meanStdDev(diff)
    low_thresh = max(10, int(mean + 0.5 * std))  # ignore very subtle differences
    high_thresh = 255
    _, thresh = cv2.threshold(diff, low_thresh, high_thresh, cv2.THRESH_BINARY)

    # Edge detection for scratches
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # Combine thresholded defects and edges
    combined = cv2.bitwise_or(thresh, edges)

    # Morphology cleanup to remove noise
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    defects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:  # smaller threshold to catch subtle defects
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        defects.append((x, y, w, h))
        # Draw white bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

    return img, defects

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="AI Leather Defect Detection Tool")
st.title("AI Leather Defect Detection Tool")
st.write("Upload, Capture, or Paste an image to inspect for defects.")

option = st.radio("Choose input method:", ["Upload Image", "Capture from Camera", "Paste Image"])
image = None

# -------------------------------
# Upload Image
# -------------------------------
if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# -------------------------------
# Camera Input
# -------------------------------
elif option == "Capture from Camera":
    camera_image = st.camera_input("Capture Image")
    if camera_image is not None:
        bytes_data = camera_image.getvalue()
        file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# -------------------------------
# Paste Image
# -------------------------------
elif option == "Paste Image":
    pasted = paste_image_button("📋 Paste Image")
    if pasted.image_data is not None:
        pil_image = pasted.image_data
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# -------------------------------
# Run Detection
# -------------------------------
if image is not None:
    annotated_image, defects = detect_defects_and_annotate(image)

    st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
             caption="Annotated Image",
             use_column_width=True)

    st.markdown(f"### 🧪 {len(defects)} defect(s) found")

    for idx, (x, y, w, h) in enumerate(defects, 1):
        st.write(f"**Defect {idx}:** Location: (x={x}, y={y}) | Size: {w}x{h} | Area: {w*h}")
