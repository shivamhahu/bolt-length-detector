from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
PROCESSED_FOLDER = os.path.join('static', 'processed')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Length estimation using OpenCV
def estimate_length(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur + Edge Detection
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 30, 100)

    # Dilation to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edged, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lengths = []
    drawn_boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        # Filter only very small noise (allow overlaps!)
        if area > 500 and w > 30 and h > 10:
            lengths.append(w)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            drawn_boxes.append((x, y, w, h))

    # Save processed image
    filename = os.path.basename(image_path)
    processed_path = os.path.join(PROCESSED_FOLDER, filename)
    cv2.imwrite(processed_path, image)

    return processed_path, lengths





@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Process image and get result path + object lengths
    result_path, lengths = estimate_length(filepath)

    # Extract just the filename
    processed_filename = os.path.basename(result_path)

    return f"""
    <h3>Detected {len(lengths)} object(s)</h3>
    Estimated widths (in pixels): {lengths}<br><br>
    <img src='/static/processed/{processed_filename}' width='400'><br><br>
    <a href='/'>Go back</a>
    """


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Render sets this dynamically
    app.run(host='0.0.0.0', port=port)
