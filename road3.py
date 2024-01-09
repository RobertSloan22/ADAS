from flask import Flask, render_template, Response
import cv2
import numpy as np
from gui_buttons import Buttons

app = Flask(__name__)


# Constants for vehicle width and camera focal length
REAL_WIDTH = 6
FOCAL_LENGTH = 700

# Function to calculate distance to camera
def distance_to_camera(known_width, focal_length, perceived_width):
    return (known_width * focal_length) / perceived_width

# Region of Interest Function
def region_of_interest(img):
    height, width = img.shape[:2]
    polygons = np.array([
        [(0, height), (width, height), (width // 2, height // 2)]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Initialize video capture and buttons
cap = cv2.VideoCapture(0)
button = Buttons()
button.add_button("person", 0, 0)
button.add_button("truck", 100, 0)
button.add_button("car", 200, 0)


colors = button.colors

net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1 / 255)

classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

print("Objects list")
print(classes)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

def click_button(event, x, y, flags, params):
    global button_person
    if event == cv2.EVENT_LBUTTONDOWN:
        button.button_click(x, y)

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Frame", click_button)
@app.route('/')

def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
def gen_frames(): 
    while True:
        ret, frame = cap.read()
      
        active_buttons = button.active_buttons_list()
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Updated HSV range for better lane detection
        lower_y = np.array([20, 100, 100])  # Adjust these values based on your testing
        upper_y = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower_y, upper_y)

        # Apply Canny edge detection
        edges = cv2.Canny(mask, 50, 150)

        # Apply Region of Interest to focus on lane lines
        roi_edges = region_of_interest(edges)

        # Detecting lines using Hough Transform
        lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, 100, minLineLength=40, maxLineGap=5)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

        # Object detection and distance calculation
        active_buttons = button.active_buttons_list()
        (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.3, nmsThreshold=0.4)
        for class_id, score, bbox in zip(class_ids, scores, bboxes):
            (x, y, w, h) = bbox
            class_name = classes[class_id]
            color = colors[class_id]

            if class_name in active_buttons:
                cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # Distance calculation for detected vehicles and objects
                if class_name in ["person", "car", "truck", "bus", "motorbike", "stop sign"]:
                    perceived_width = w
                    distance_in_feet = distance_to_camera(REAL_WIDTH, FOCAL_LENGTH, perceived_width)
                    
                    if distance_in_feet < 50:  # Adjust this threshold as needed
                        alert_message = "{} is {} feet away".format(class_name, round(distance_in_feet, 2))
                        cv2.putText(frame, alert_message, (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        button.display_buttons(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
