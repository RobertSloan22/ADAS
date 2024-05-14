from flask import Flask, render_template, Response
import jetson.inference
import jetson.utils
import numpy as np
import cv2
import math

app = Flask(__name__)

# Load the SSD MobileNet model
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

# Constants for distance calculation
REAL_WIDTH = 78  # Known width of the object in inches (6.5 feet = 78 inches)
KNOWN_WIDTH = 78
FOCAL_LENGTH = 720
PROXIMITY_ALERT_DISTANCE = 15 * 12  # 15 feet in inches

# Load reference image and calculate focal length
ref_image = cv2.imread("logitech.png")
ref_distance = 240  # 20 feet = 240 inches
ref_width = 78  # 6.5 feet = 78 inches
focal_length = (ref_distance * ref_width) / REAL_WIDTH

def calculate_distance(known_width, focal_length, perceived_width):
    """Calculate distance from camera to object."""
    return (known_width * focal_length) / perceived_width

# Flask route to serve the main page
@app.route('/')
def index():
    # Ensure 'index4.html' is in the templates directory
    return render_template('index4.html')

# Flask route to serve video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    # Create a video source object for the USB camera (adjust the device number if needed)
    camera = jetson.utils.videoSource("v4l2:///dev/video0")  # Update device path if different

    while True:
        img = camera.Capture()
        detections = net.Detect(img)
        # Convert the image to an array format that can be used with OpenCV
        frame = jetson.utils.cudaToNumpy(img)
        
        proximity_alert = False
        
        # Draw detections on the image
        for detection in detections:
            ID = detection.ClassID
            top = int(detection.Top)
            left = int(detection.Left)
            bottom = int(detection.Bottom)
            right = int(detection.Right)
            jetson.utils.cudaDrawRect(img, (left, top, right, bottom), (255, 0, 0, 150))
            
            # Calculate the perceived width of the detected object
            perceived_width = right - left
            # Calculate the distance to the detected object
            distance = calculate_distance(KNOWN_WIDTH, focal_length, perceived_width)
            
            # Check for proximity alert
            if distance <= PROXIMITY_ALERT_DISTANCE:
                proximity_alert = True
            
            # Draw the distance on the frame
            cv2.putText(frame, f"Distance: {distance:.2f} in", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Flip the frame vertically and horizontally
        frame = cv2.flip(frame, -1)  # -1 means flipping around both axes

        # Draw proximity alert if needed
        if proximity_alert:
            alert_text = "PROXIMITY ALERT!"
            font_scale = 1.5
            font_thickness = 4
            text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] + text_size[1]) // 2
            cv2.putText(frame, alert_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)
            
            # Draw the red rectangle around the text
            rect_start = (text_x - 10, text_y - text_size[1] - 10)
            rect_end = (text_x + text_size[0] + 10, text_y + 10)
            cv2.rectangle(frame, rect_start, rect_end, (0, 0, 255), 2)

        # Encode the frame in JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        # Yield the frame data
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, threaded=True)
