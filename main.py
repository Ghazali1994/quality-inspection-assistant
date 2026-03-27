import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_paste_button import paste_image_button


def detect_defects_and_annotate(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to highlight dark defects
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Remove small noise
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours (exact defect shapes)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    defects = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # ignore tiny noise
        if area > 100:
            defects.append(cnt)

            # draw exact contour (WHITE)
            cv2.drawContours(
                image,
                [cnt],
                -1,
                (255, 255, 255),  # white
                2
            )

    return image, defects


# UI
st.set_page_config(page_title="AI Leather Defect Detection Tool")
st.title("AI Leather Defect Detection Tool")
st.write("Upload, Capture, or Paste an image to inspect for defects.")

# selector
option = st.radio(
    "Choose input method:",
    ["Upload Image", "Capture from Camera", "Paste Image"]
)

image = None

# Upload
if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# Camera
elif option == "Capture from Camera":
    camera_image = st.camera_input("Capture Image")
    if camera_image is not None:
        bytes_data = camera_image.getvalue()
        file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# Paste
elif option == "Paste Image":
    pasted = paste_image_button("📋 Paste Image")
    if pasted.image_data is not None:
        pil_image = pasted.image_data
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


# Run detection
if image is not None:
    annotated_image, defects = detect_defects_and_annotate(image.copy())

    st.image(
        cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
        caption="Annotated Image",
        use_column_width=True
    )

    st.markdown(f"### 🧪 {len(defects)} defect(s) found")

    # show defect info
    for idx, cnt in enumerate(defects, 1):
        area = int(cv2.contourArea(cnt))
        x, y, w, h = cv2.boundingRect(cnt)

        st.write(
            f"**Defect {idx}:** Area: {area} px | Location: (x={x}, y={y}) | Size: {w}x{h}"
        )
```
