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
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lengths = []
    drawn_boxes = []  # List to store the coordinates of the drawn boxes

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter small objects by width and height
        if w > 50 and h > 10:
            overlap_found = False

            # Check for overlap with already drawn boxes
            for (prev_x, prev_y, prev_w, prev_h) in drawn_boxes:
                # If the current box overlaps with a previous box, don't draw it again
                if (x < prev_x + prev_w and x + w > prev_x and 
                    y < prev_y + prev_h and y + h > prev_y):
                    overlap_found = True
                    break
            
            # If no overlap is found, draw the box and add it to the list of drawn boxes
            if not overlap_found:
                lengths.append(w)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                drawn_boxes.append((x, y, w, h))

    # Save the processed image
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
