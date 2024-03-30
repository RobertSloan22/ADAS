from flask import Flask, render_template, Response
import cv2
import math
import numpy as np
from gui_buttons import Buttons
from threading import Thread



app = Flask(__name__)

# Define colors
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
# Constants for vehicle width and camera focal length
REAL_WIDTH = 6
KNOWN_WIDTH = 6
FOCAL_LENGTH = 720
# Load reference image and calculate focal length
ref_image = cv2.imread("logitech.png")
gray_ref = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
car_cascade = cv2.CascadeClassifier("cars1.xml")
cars = car_cascade.detectMultiScale(gray_ref, 1.1, 5)
ref_distance = 240  # 20 feet = 240 inchespip install flask opencv-python-headless
ref_width = 78  # 6.5 feet = 78 inches
focal_length = (ref_distance * ref_width) / cars[0][2]
def calculate_distance(point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def click_button(event, x, y, flags, params):
    global button_person
    if event == cv2.EVENT_LBUTTONDOWN:
        button.button_click(x, y)

# Function to calculate distance to camera
def distance_to_camera(known_width, focal_length, perceived_width):
    return (known_width * focal_length) / perceived_width

def calculate_distance(point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

class LaneDetector:
    def __init__(self):
        self.previous_lines = []

# Region of Interest Function
def make_coordinates(image, line_parameters, side):
    try:
        slope, intercept = line_parameters
        # Check that slope and intercept are within reasonable bounds
        if abs(slope) > 1e9 or abs(intercept) > 1e9:
            raise ValueError("Slope or intercept is too large")
        # Check that slope is not too small to avoid division by zero
        if abs(slope) < 1e-5:
            raise ValueError("Slope is too small")
    except (TypeError, ValueError) as e:
        print(f"Unexpected line_parameters: {line_parameters}, error: {str(e)}")
        # Set default slope and intercept
        slope = 0.5 if side == 'left' else -0.5
        intercept = image.shape[0]
    y1 = image.shape[0]
    y2 = int(y1*(14/20))  # Adjust this line to change the length of the line
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2], dtype=int)

def averaged_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope > 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average, 'left')
    right_line = make_coordinates(image, right_fit_average, 'right')
    return np.array([left_line, right_line])

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (0,255, 0), 25)
    return line_image

def region_of_interest(image):
    height, width = image.shape[:2]
    triangle = np.array([
       [(200, 510), (850, 530), (476, 296)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image




# Initialize video capture and buttons
# Initialize video capture andr buttons
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)


#Active buttons for class detection
button = Buttons()
button.add_button("person", 0, 0)
button.add_button("truck", 200, 0)
button.add_button("car", 400, 0)
button.add_button("bus", 600, 0)

# Define the location for where the alert message should be displayed in the upper left hand corner of the frame



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

def click_button(event, x, y, flags, params):
    global button_person
    if event == cv2.EVENT_LBUTTONDOWN:
        button.button_click(x, y)

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Frame", click_button)
@app.route('/')

def index():
    return render_template('index1.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
def gen_frames(): 
    while True:
        ret, frame = cap.read()
        _, frame = cap.read()
        frame = cv2.flip(frame, -1)
        combo_image = frame
        canny_image = canny(frame)
        cropped_image = region_of_interest(canny_image)
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=10, maxLineGap=5)
        if lines is not None:
            averaged_lines = averaged_slope_intercept(frame, lines)
            if averaged_lines is not None and all(line is not None for line in averaged_lines):
                line_image = display_lines(frame, averaged_lines)
                combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

                # Replace 'frame' with 'combo_image' in the following line
          
        # Object detection and distance calculation
        active_buttons = button.active_buttons_list()
        (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.3, nmsThreshold=0.4)
        for class_id, score, bbox in zip(class_ids, scores, bboxes):
            (x, y, w, h) = bbox
            class_name = classes[class_id]
            color = colors[class_id]
             # Calculate distances and store them with associated data
            distances = []
            for class_id, score, bbox in zip(class_ids, scores, bboxes):
                (x, y, w, h) = bbox
                center_x, center_y = x + w / 2, y + h / 2  # Calculate center of bbox
                distance = np.sqrt(center_x**2 + center_y**2)  # Calculate distance from top-left corner
                distances.append((distance, class_id, score, bbox))
            distances.sort()
            nearest_cars = distances[:6]  # Get the three nearest cars
            # Display the distance to all detected cars in the upper left hand corner of the frame
         
            # Process the nearest cars
            for distance, class_id, score, bbox in nearest_cars:
                (x, y, w, h) = bbox
                class_name = classes[class_id]
                color = colors[class_id]

            if class_name in active_buttons:
                cv2.putText(frame, class_name, (x, y - 15), cv2.FONT_HERSHEY_DUPLEX, 1, BLUE, 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), BLUE, 3)
                
                
                center = (int(x + w/2), int(y + h/2))
                radius = int(math.sqrt((w/2)**2 + (h/2)**2))
                # Draw the bounding circle
                cv2.circle(frame, center, radius, BLUE, thickness=4)
                # Draw lines emanating from the center of the circle
               # for angle in range(0, 360, 90):
                    #end_point_x = int(center[0] + radius * math.cos(math.radians(angle)))
                    #end_point_y = int(center[1] + radius * math.sin(math.radians(angle)))
                   # cv2.line(frame, center, (end_point_x, end_point_y), color, thickness=3)
                
                # Distance calculation for detected vehicles and objects
                if class_name in ["person", "car", "truck", "bus", "motorbike", "stop sign"]:
                    perceived_width = w
                    distance_in_feet = distance_to_camera(REAL_WIDTH, FOCAL_LENGTH, perceived_width)
                    
                    if distance_in_feet < 50:  # Adjust this threshold as needed
                        distance_message = "DETECTION: {} is {} feet away".format(class_name, round(distance_in_feet, 2))
                        cv2.putText(frame, distance_message, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                        if distance_in_feet < 15:
                            alert_message = "PROXIMITY ALERT".format(class_name, round(distance_in_feet, 2))
                            cv2.putText(frame, alert_message, (int(frame.shape[1] / 2 - 150), int(frame.shape[0] / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                            cv2.rectangle(frame, (int(frame.shape[1] / 2 - 150), int(frame.shape[0] / 2 - 75)), (int(frame.shape[1] / 2 + 150), int(frame.shape[0] / 2 + 75)), (0, 0, 255), 2)        
                            # Print the alert message to the console and on the left hand side of the top of the frame 
                            print(alert_message)
                            #cv2.putText(frame, alert_message, alert_message_location, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            
                
        fused_image = cv2.addWeighted(combo_image, 0.6, frame, 0.4, 0)
        
    # Replace 'frame' with 'fused_image' in the following line
        ret, buffer = cv2.imencode('.jpg', fused_image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
