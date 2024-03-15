from flask import Flask, render_template, Response
import cv2
import math
import numpy as np
from gui_buttons import Buttons
import threading

# Initialize Flask app
app = Flask(__name__)

# Define constants and colors
REAL_WIDTH = 6  # Vehicle width in feet for distance estimation
FOCAL_LENGTH = 720  # Approximate focal length of the camera
RED, GREEN, BLUE = (0, 0, 255), (0, 255, 0), (255, 0, 0)

# Load your reference image and car cascade here if needed for initial setup

# Function placeholders for your existing logic (e.g., distance calculations)

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Flask route to serve the main page
@app.route('/')
def index():
    return render_template('index.html')

# Flask route to serve video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames(): 
    # Initialize video capture
    cap = cv2.VideoCapture(0)

    # Setup for CUDA-accelerated DNN
    net = cv2.dnn.readNetFromDarknet("dnn_model/yolov4-tiny.cfg", "dnn_model/yolov4-tiny.weights")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Preprocessing for object detection
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward(output_layers)

        # Post-processing detections
        for detection in detections:
            # Your logic to handle each detection, including:
            # - Extracting bounding box coordinates
            # - Calculating distances
            # - Drawing bounding boxes and annotations on `frame`
            pass  # Replace with actual processing logic

        # Encode frame for HTTP response
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            continue

        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
