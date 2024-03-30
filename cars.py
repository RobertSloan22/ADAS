from flask import Flask, render_template, Response
import cv2
import math
import numpy as np

app = Flask(__name__)

# Constants
KNOWN_WIDTH = 6  # Average width of a car in feet
FOCAL_LENGTH = 720  # Focal length, adjust based on your camera setup

# Load the Haar cascade for car detection
car_cascade = cv2.CascadeClassifier('cars1.xml')

def distance_to_camera(known_width, focal_length, perceived_width):
    """
    Calculates the distance from the camera to an object based on its perceived width in the image.
    :param known_width: The known width of the object (in the same units as the focal length).
    :param focal_length: The focal length of the camera (in the same units as the known width).
    :param perceived_width: The width of the object as observed in the image (pixels).
    :return: The distance from the object to the camera.
    """
    return (known_width * focal_length) / perceived_width

def calculate_distance(point1, point2):
    """
    Calculates the Euclidean distance between two points in 2D space.
    :param point1: First point as a tuple (x1, y1).
    :param point2: Second point as a tuple (x2, y2).
    :return: Euclidean distance between the two points.
    """
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

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
    y2 = int(y1*(14/20))  # Adjust this line to change the length of the line
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2], dtype=int)

def averaged_slope_intercept(image, lines):
    if lines is None:
        return None
    for line in lines:
        left_fit = []
        right_fit = []
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
       [(360, 540), (850, 530), (440, 205)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

@app.route('/')
def index():
    """Renders the main page."""
    return render_template('index3.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    """Generates video frames."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
# Use your video source
    while True:
        success, frame = cap.read()
        ret, frame = cap.read()
        _, frame = cap.read()
        frame = cv2.flip(frame, -1)

        if not success:
            break

        # Apply canny and region_of_interest to limit car detection to ROI
        canny_image = canny(frame)
       # roi_canny = region_of_interest(canny_image)
        # Use ROI image for car detection
        cars = car_cascade.detectMultiScale(frame, 1.4, 1)
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 4)
            perceived_width = w
            distance_in_feet = distance_to_camera(KNOWN_WIDTH, FOCAL_LENGTH, perceived_width)
            cv2.putText(frame, f"{distance_in_feet:.2f} ft", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Lane detection continues as before
        cropped_canny = region_of_interest(canny_image)
        lines = cv2.HoughLinesP(cropped_canny, 2, np.pi / 180, 100, np.array([]), minLineLength=20, maxLineGap=5)
        averaged_lines = averaged_slope_intercept(frame, lines)
        line_image = display_lines(frame, averaged_lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

     
        ret, buffer = cv2.imencode('.jpg', combo_image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
