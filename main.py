from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, RedirectResponse
import cv2
import numpy as np
import os
import uuid

app = FastAPI()

ANNOTATED_DIR = "annotated_images"
os.makedirs(ANNOTATED_DIR, exist_ok=True)

@app.get("/")
def redirect_to_upload():
    return RedirectResponse(url="/upload")


@app.get("/upload", response_class=HTMLResponse)
def upload_form():
    return """
    <html>
    <head>
        <title>Quality Inspection Assistant - Upload</title>
        <style>
            body {
                font-family: 'Segoe UI', sans-serif;
                background-color: #f4f6f9;
                padding: 30px;
                text-align: center;
            }
            h2 {
                color: #2c3e50;
            }
            form {
                margin-top: 20px;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                display: inline-block;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            input[type="file"] {
                margin: 10px 0;
                font-size: 16px;
            }
            input[type="submit"] {
                background-color: #3498db;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
            }
            input[type="submit"]:hover {
                background-color: #2980b9;
            }
            a.back-link {
                display: block;
                margin-top: 20px;
                color: #3498db;
                text-decoration: none;
            }
            a.back-link:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <h2>🛡️ Quality Inspection Assistant</h2>
        <form action="/inspect" enctype="multipart/form-data" method="post">
            <input type="file" name="file" accept="image/*" required>
            <br><br>
            <input type="submit" value="Inspect Image">
        </form>
    </body>
    </html>
    """


def detect_defects_and_annotate(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    defects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            defects.append({"x": x, "y": y, "width": w, "height": h, "area": area})
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return image, defects


@app.post("/inspect")
async def inspect_image(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image file"})

    annotated_img, defects = detect_defects_and_annotate(img)

    unique_filename = f"{uuid.uuid4().hex}.jpg"
    save_path = os.path.join(ANNOTATED_DIR, unique_filename)
    cv2.imwrite(save_path, annotated_img)

    defect_list_html = ''.join([
        f"<li>Area: {int(d['area'])}, Position: (x={d['x']}, y={d['y']}), Size: {d['width']}x{d['height']}</li>"
        for d in defects
    ])

    return HTMLResponse(content=f"""
    <html>
    <head>
        <title>Inspection Results</title>
        <style>
            body {{
                font-family: 'Segoe UI', sans-serif;
                background-color: #f9fafb;
                padding: 30px;
                text-align: center;
            }}
            .card {{
                background-color: white;
                padding: 25px;
                border-radius: 8px;
                display: inline-block;
                box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                max-width: 700px;
            }}
            h3 {{
                color: #2c3e50;
            }}
            ul {{
                text-align: left;
                display: inline-block;
                margin-top: 10px;
            }}
            img {{
                max-width: 100%;
                border: 1px solid #ccc;
                margin-top: 20px;
                border-radius: 4px;
            }}
            a {{
                display: inline-block;
                margin-top: 20px;
                color: #3498db;
                text-decoration: none;
                font-weight: bold;
            }}
            a:hover {{
                text-decoration: underline;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <h3>🧪 Inspection Results</h3>
            <p><strong>Defects found:</strong> {len(defects)}</p>
            <ul>{defect_list_html}</ul>
            <img src="/annotated/{unique_filename}" alt="Annotated Image">
            <br><a href="/upload">🔙 Inspect Another Image</a>
        </div>
    </body>
    </html>
    """)


@app.get("/annotated/{filename}", response_class=FileResponse)
def get_annotated_image(filename: str):
    file_path = os.path.join(ANNOTATED_DIR, filename)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "Image not found"})
    return FileResponse(file_path, media_type="image/jpeg")
