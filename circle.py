from flask import Flask, render_template, Response
import cv2
import math
import numpy as np
from gui_buttons import Buttons
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
lane_model = load_model('model.h5')

# Define colors
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

# Constants for vehicle width and camera focal length
REAL_WIDTH = 6
FOCAL_LENGTH = 720
def preprocess_frame_for_lane_detection(frame):
    # Adjust the preprocessing based on your model's needs
    resized_frame = cv2.resize(frame, (160, 80))  # Example resize to model's input size
    normalized_frame = resized_frame / 255.0  # Normalizing
    return np.array([normalized_frame])  # Adding batch dimension

def draw_lanes(original_image, predictions):
    # Assuming the predictions are a mask of the lanes
    # Adjust the postprocessing based on your model's output
    
    # Example of processing a mask output to draw lanes
    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            if predictions[i, j] > 0.5:  # Thresholding prediction
                cv2.circle(original_image, (j*4, i*4), 1, GREEN, -1)  # Scale up to original size and draw

    return original_image
def distance_to_camera(known_width, focal_length, perceived_width):
    return (known_width * focal_length) / perceived_width

def calculate_distance(point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def region_of_interest(img, vertices):
    
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

cap = cv2.VideoCapture('solidwR.mp4')
button = Buttons()
button.add_button("person", 0, 0)
button.add_button("truck", 200, 0)
button.add_button("car", 400, 0)
button.add_button("bus", 600, 0)

colors = button.colors


net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1 / 255)

classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

def click_button(event, x, y, flags, params):
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

        while(cap.isOpened()):
            ret, frame = cap.read()

            if ret:  # Check if frame was successfully read
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blur, 50, 150)

                # Define the vertices for the region of interest
                height = frame.shape[0]
                vertices = np.array([[(100, height), (1644, height), (760,230)]])
                
                roi = region_of_interest(edges, vertices)
                # Rest of your code...
         # Exit the loop if no frame was read
        # Rest of your processing and object detection

 
        # Apply Gaussian blur
        #frame = cv2.GaussianBlur(frame, (5, 5), 0)
        #hsv
        frame = cv2.flip(frame, 1)
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Preprocess frame for lane model
        preprocessed_frame = preprocess_frame_for_lane_detection(frame)
        
        # Predict lanes
        lane_predictions = lane_model.predict(preprocessed_frame)[0]
        
        # Draw lanes on the frame
        frame = draw_lanes(frame, lane_predictions)
        # Convert the image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blur, 50, 150)

        # Apply the region of interest mask
        roi = region_of_interest(edges)

        # Convert the original image to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define color ranges for white and yellow
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 255, 255])
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        # Create color masks
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Combine the color masks
        color_mask = cv2.bitwise_or(white_mask, yellow_mask)

        # Combine the color mask and the edges mask
        final_mask = cv2.bitwise_and(roi, color_mask)

        
        # Apply the final mask to the original image
        result = cv2.bitwise_and(frame, frame, mask=final_mask)

        # Display the result
        cv2.imshow('Result', result)
        # Define the lower and upper bounds of the HSV range
        #lower_y = np.array([88 - 10, 33 - 10, 211 - 10])
       # upper_y = np.array([88 + 10, 33 + 10, 211 + 10])
        #mask = cv2.inRange(hsv, lower_y, upper_y)

        # Create a mask
       # mask = cv2.inRange(hsv, lower_y, upper_y)
                
        # Apply Canny edge detection
       # edges = cv2.Canny(mask, 74, 150)
        
        # Apply Region of Interest to focus on lane lines
       # roi_edges = region_of_interest(edges)
      

        # Detecting lines using Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, maxLineGap=50)
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
                cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, BLUE, 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), BLUE, 2)
                
                
                center = (int(x + w/2), int(y + h/2))
                radius = int(math.sqrt((w/2)**2 + (h/2)**2))
                # Draw the bounding circle
                cv2.circle(frame, center, radius, BLUE, thickness=2)
                # Draw lines emanating from the center of the circle
                for angle in range(0, 360, 90):
                    end_point_x = int(center[0] + radius * math.cos(math.radians(angle)))
                    end_point_y = int(center[1] + radius * math.sin(math.radians(angle)))
                    cv2.line(frame, center, (end_point_x, end_point_y), color, thickness=1)
                
                # Distance calculation for detected vehicles and objects
                if class_name in ["person", "car", "truck", "bus", "motorbike", "stop sign"]:
                    perceived_width = w
                    distance_in_feet = distance_to_camera(REAL_WIDTH, FOCAL_LENGTH, perceived_width)
                    
                    if distance_in_feet < 50:  # Adjust this threshold as needed
                        alert_message = "{} is {} feet away".format(class_name, round(distance_in_feet, 2))
                        cv2.putText(frame, alert_message, (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        if distance_in_feet < 10:
                            alert_message = "{} is {} FT".format(class_name, round(distance_in_feet, 2))
                            cv2.putText(frame, alert_message, (int(frame.shape[1] / 2 - 150), int(frame.shape[0] / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                            cv2.rectangle(frame, (int(frame.shape[1] / 2 - 150), int(frame.shape[0] / 2 - 75)), (int(frame.shape[1] / 2 + 150), int(frame.shape[0] / 2 + 75)), (0, 0, 255), 2)        
        
    
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)