import os
import cv2
import numpy as np
from ultralytics import YOLO

# Define the model path
model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')

# Load the model
model = YOLO('/home/adithyadk/Desktop/model-ai/trained-model/seg-model/best(seg).pt')

# Define the threshold value
threshold = 0.1

# Open the camera
cap = cv2.VideoCapture(0)

# Set the frame rate to 15 fps
cap.set(cv2.CAP_PROP_FPS, 15)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Perform object detection
    results = model(frame)[0]

    # Iterate over the detections
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        # Check if the score is above the threshold
        if score > threshold:
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Segment the detected object by applying a color filter to the bounding box area
            segmented_object = frame[int(y1):int(y2), int(x1):int(x2)]
            segmented_object = cv2.inRange(segmented_object, (0, 0, 100), (255, 255, 255))
            segmented_object = cv2.bitwise_and(segmented_object, segmented_object, mask=segmented_object)

            # Calculate the centroid of the segmented object
            moments = cv2.moments(segmented_object)
            if moments['m00'] != 0:
                centroid_x = int(moments['m10'] / moments['m00'])
                centroid_y = int(moments['m01'] / moments['m00'])

            # Draw a circle at the centroid
            cv2.circle(frame, (int(x1 + centroid_x), int(y1 + centroid_y)), 5, (0, 0, 255), -1)

    # Display the output frame
    cv2.imshow("Image with Detections", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()


























































































































































































































































































































