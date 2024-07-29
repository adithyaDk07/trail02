import os
import cv2
import numpy as np
from ultralytics import YOLO

# Define
image_path = '/home/adithyadk/Desktop/model-ai/dataset/dataset4/010101.v1i.yolov9/test/images/WhatsApp-Image-2024-07-29-at-9-52-30-AM_jpeg.rf.7d8713bc18ce758726338515344d12cc.jpg'
image = cv2.imread(image_path)

# Define the model path
model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')

# Load the object detection model
model = YOLO(r"/home/adithyadk/Desktop/model-ai/trained-model/seg-model/best(seg).pt")

# Define the threshold value
threshold = 0.5

# Perform object detection
results = model(image)[0]

# Convert the image to HSV color space
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Iterate over the detections
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    # Check if the score is above the threshold
    if score > threshold:
        # Draw bounding box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Segment the detected object using color segmentation
        segmented_object_hsv = image_hsv[int(y1):int(y2), int(x1):int(x2)]
        # Apply thresholding based on HSV values (e.g., segment objects with a specific color range)
        segmented_mask = cv2.inRange(segmented_object_hsv, (0, 50, 50), (180, 255, 255))

        # Apply the segmented mask to the original image
        segmented_object = cv2.bitwise_and(segmented_object_hsv, segmented_object_hsv, mask=segmented_mask)

        # Calculate the centroid of the segmented object
        moments = cv2.moments(segmented_mask)
        centroid_x = int(moments['m10'] / moments['m00'])
        centroid_y = int(moments['m01'] / moments['m00'])

        # Draw a circle at the centroid
        cv2.circle(image, (int(x1 + centroid_x), int(y1 + centroid_y)), 5, (0, 0, 255), -1)

# Define the output image path
output_image_path = '/home/adithyadk/Desktop/model-ai/output_image.jpg'

# Save the output image
cv2.imwrite(output_image_path, image)

# Display the output image
cv2.imshow("Image with Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()