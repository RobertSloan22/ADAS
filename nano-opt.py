from flask import Flask, render_template, Response
import cv2
import math
import numpy as np
from threading import Thread
import jetson.inferece
import jetson.utils



app = Flask(__name__)

# Constants for vehicle width and camera focal length
REAL_WIDTH = 6
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
        if abs(slope) > 1e5 or abs(intercept) > 1e5:
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
    y2 = int(y1*(15/20))  # Adjust this line to change the length of the line
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
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 50)
    return line_image


def region_of_interest(image):
    height, width = image.shape[:2]
    triangle = np.array([
       [(150, 690), (1100, 690), (750, 350)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
cap = jetson.utils.gstCamera(1280, 720, "/dev/video1")

display = jetson.utils.glDisplay()

@app.route('/')

def index():
    return render_template('index1.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
def gen_frames(): 
    while True:
        img, width, height = cap.CaptureRGBA()
        detections = net.Detect(img, width, height) 
        display.RenderOnce(img, width, height)
        display.SetTitle("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
        frame = cap.read()
        _, frame = cap.read()
        frame = cv2.flip(frame, -1)
        combo_image = frame
        canny_image = canny(frame)
        cropped_image = region_of_interest(canny_image)
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=20)
        if lines is not None:
            averaged_lines = averaged_slope_intercept(frame, lines)
            if averaged_lines is not None and all(line is not None for line in averaged_lines):
                line_image = display_lines(frame, averaged_lines)
                combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
                cv2.imshow("result", combo_image)


        fused_image = cv2.addWeighted(combo_image, 0.4, frame, 0.6, 0)
        
    # Replace 'frame' with 'fused_image' in the following line
        ret, buffer = cv2.imencode('.jpg', fused_image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
