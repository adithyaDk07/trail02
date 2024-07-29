import os
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = '/home/adithyadk/Desktop/model-ai/dataset/dataset4/010101.v1i.yolov9/train/images/WhatsApp-Image-2024-07-29-at-9-52-26-AM_jpeg.rf.494488a0d27b1b0b572dfaebb756cde5.jpg'  # Replace with the path to your image
image = cv2.imread(image_path)

model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')

# Load a model
model = YOLO('/home/adithyadk/Desktop/model-ai/trained-model/best.pt')  # load a custom model

threshold = 0.5


results = model(image)[0]

for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        # Define the polygon points for the bounding box
        points = np.array([[(int(x1), int(y1)), (int(x2), int(y1)), (int(x2), int(y2)), (int(x1), int(y2))]])

        # Define the color for the labeled box
        color = (0, 255, 0)  # Green color

        # Fill the polygon with the specified color
        cv2.fillPoly(image, points, color)

output_image_path = '/home/adithyadk/Desktop/model-ai/output_image.jpg' # Replace with the desired output image path
cv2.imwrite(output_image_path, image)


cv2.imshow("Image with Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()