import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_paste_button import paste_image_button


# --------------------------------
# Industrial Defect Detection
# --------------------------------
def detect_defects_and_annotate(image):
    img = image.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Smooth texture
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # -------- LIGHT DEFECTS ----------
    light = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        -3
    )

    # -------- DARK DEFECTS ----------
    dark = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        3
    )

    # -------- EDGE DEFECTS (SCRATCH) ----------
    edges = cv2.Canny(blur, 40, 120)

    # Combine all
    combined = cv2.bitwise_or(light, dark)
    combined = cv2.bitwise_or(combined, edges)

    # Morphology cleanup
    kernel = np.ones((3, 3), np.uint8)

    morph = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    morph = cv2.morphologyEx(
        morph,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # Find contours
    contours, _ = cv2.findContours(
        morph,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    defects = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Remove tiny noise
        if area < 120:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        aspect_ratio = w / float(h)

        # Filter unrealistic shapes
        if 0.05 < aspect_ratio < 15:
            defects.append((x, y, w, h))

            # WHITE BOX
            cv2.rectangle(
                img,
                (x, y),
                (x + w, y + h),
                (255, 255, 255),
                2
            )

    return img, defects


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="AI Leather Defect Detection Tool")
st.title("AI Leather Defect Detection Tool")
st.write("Upload, Capture, or Paste an image to inspect for defects.")

option = st.radio(
    "Choose input method:",
    ["Upload Image", "Capture from Camera", "Paste Image"]
)

image = None


# Upload
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


# Camera
elif option == "Capture from Camera":
    camera_image = st.camera_input("Capture Image")

    if camera_image is not None:
        bytes_data = camera_image.getvalue()

        file_bytes = np.asarray(
            bytearray(bytes_data),
            dtype=np.uint8
        )

        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


# Paste
elif option == "Paste Image":
    pasted = paste_image_button("📋 Paste Image")

    if pasted.image_data is not None:
        pil_image = pasted.image_data

        image = cv2.cvtColor(
            np.array(pil_image),
            cv2.COLOR_RGB2BGR
        )


# Run detection
if image is not None:
    annotated_image, defects = detect_defects_and_annotate(image)

    st.image(
        cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
        caption="Annotated Image",
        use_column_width=True
    )

    st.markdown(f"### 🧪 {len(defects)} defect(s) found")

    for idx, (x, y, w, h) in enumerate(defects, 1):
        st.write(
            f"**Defect {idx}:** "
            f"Location: (x={x}, y={y}) | "
            f"Size: {w}x{h} | "
            f"Area: {w*h}"
        )
