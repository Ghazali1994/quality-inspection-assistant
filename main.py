```python
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_paste_button import paste_image_button


def detect_defects_and_annotate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # smooth leather texture
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # adaptive threshold (better for uneven leather)
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    # join broken defect regions
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    defects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # lower = more sensitive
        if area > 50:
            x, y, w, h = cv2.boundingRect(cnt)
            defects.append((x, y, w, h))

            # WHITE BOX
            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (255, 255, 255),
                2
            )

    return image, defects


st.set_page_config(page_title="AI Leather Defect Detection Tool")
st.title("AI Leather Defect Detection Tool")
st.write("Upload, Capture, or Paste an image to inspect for defects.")

option = st.radio(
    "Choose input method:",
    ["Upload Image", "Capture from Camera", "Paste Image"]
)

image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

elif option == "Capture from Camera":
    camera_image = st.camera_input("Capture Image")
    if camera_image is not None:
        bytes_data = camera_image.getvalue()
        file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

elif option == "Paste Image":
    pasted = paste_image_button("📋 Paste Image")
    if pasted.image_data is not None:
        pil_image = pasted.image_data
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


if image is not None:
    annotated_image, defects = detect_defects_and_annotate(image.copy())

    st.image(
        cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
        caption="Annotated Image",
        use_column_width=True
    )

    st.markdown(f"### 🧪 {len(defects)} defect(s) found")

    for idx, (x, y, w, h) in enumerate(defects, 1):
        st.write(
            f"**Defect {idx}:** Location: (x={x}, y={y}), Size: {w}x{h}, Area: {w*h}"
        )
```
