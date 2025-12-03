from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os
import torch
import traceback

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ---------------------------------------------------------------
# 1. LOAD MODELS
# ---------------------------------------------------------------
person_model = YOLO("yolov8n.pt")  # YOLOv8n COCO for person detection
ppe_model = YOLO("best.pt")        # Your trained PPE model

# Auto GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
person_model.to(device)
ppe_model.to(device)
print(f"Models loaded on: {device}")

# ---------------------------------------------------------------
# 2. PPE Classes
# ---------------------------------------------------------------
CLASSES = {
    0: 'Person',
    1: 'Helmet',
    2: 'Vest',
    3: 'Boots',
    4: 'Ear-protection',
    5: 'Mask',
    6: 'Glass',
    7: 'Glove'
}

REQUIRED_PPE = ['Helmet', 'Vest']  # Change as needed

YOLO_CONF_THRESHOLD = 0.5

# ---------------------------------------------------------------
# 3. PROCESS FRAME
# ---------------------------------------------------------------
def process_ppe_frame(frame):
    h, w = frame.shape[:2]

    # Stage 1: Detect persons
    results_person = person_model(frame, imgsz=640, conf=YOLO_CONF_THRESHOLD, device=device)[0]
    persons = []
    for box in results_person.boxes:
        cls = int(box.cls[0])
        if cls == 0:  # person
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            persons.append((x1, y1, x2, y2))

    # Stage 2: Detect PPE per person
    for (px1, py1, px2, py2) in persons:
        person_crop = frame[py1:py2, px1:px2]
        if person_crop.size == 0:
            continue

        results_ppe = ppe_model(person_crop, imgsz=640, conf=YOLO_CONF_THRESHOLD, device=device)[0]

        wearing = {item: False for item in REQUIRED_PPE}
        for box in results_ppe.boxes:
            cls = int(box.cls[0])
            label = CLASSES.get(cls, "Unknown")
            if label in REQUIRED_PPE:
                wearing[label] = True

        # Determine protection
        if all(wearing.values()):
            status = "PROTECTED"
            color = (0, 255, 0)
        else:
            status = "UNPROTECTED"
            color = (0, 0, 255)

        # Draw bounding box + status
        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 3)
        cv2.putText(frame, status, (px1, py1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    return frame

# ---------------------------------------------------------------
# 4. ROUTES
# ---------------------------------------------------------------
@app.route("/")
def home():
    return jsonify({"message": "PPE Detection API Running"})

@app.route("/detect_ppe", methods=["POST"])
def detect_ppe():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov"]:
            return jsonify({"error": "Unsupported file type"}), 400

        # Temp files
        input_path = tempfile.NamedTemporaryFile(delete=False, suffix=ext).name
        output_ext = ".mp4" if ext in [".mp4", ".avi", ".mov"] else ".jpg"
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=output_ext).name
        file.save(input_path)

        # Video processing
        if ext in [".mp4", ".avi", ".mov"]:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                return jsonify({"error": "Cannot open video file"}), 400

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                processed = process_ppe_frame(frame)
                out.write(processed)

            cap.release()
            out.release()
            return send_file(output_path, mimetype="video/mp4",
                             as_attachment=True, download_name="ppe_detected.mp4")

        # Image processing
        else:
            img = cv2.imread(input_path)
            if img is None:
                return jsonify({"error": "Cannot read image file"}), 400

            result = process_ppe_frame(img)
            cv2.imwrite(output_path, result)
            return send_file(output_path, mimetype="image/jpeg",
                             as_attachment=True, download_name="ppe_detected.jpg")

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        for p in [input_path, output_path]:
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except:
                    pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)
