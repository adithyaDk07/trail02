import torch
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
import cv2

# Load pre-trained Mask R-CNN model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set to evaluation mode

# Open webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera

# Set the frame rate to 10 fps
cap.set(cv2.CAP_PROP_FPS, 10)

# Camera intrinsic parameters (example values, replace with actual values)
fx = 10447.820616852037  # Focal length in x (pixels)
fy = 10477.457878200092  # Focal length in y (pixels)
cx = 305.83263140233214  # Principal point x (pixels)
cy = 954.334079526126  # Principal point y (pixels)

# Depth of the object from the camera (in meters)
depth = 0.45  # Replace with the actual distance if known

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL Image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Preprocess the image
    image_tensor = F.to_tensor(image).unsqueeze(0)  # Convert to tensor and add batch dimension

    # Perform inference
    with torch.no_grad():
        predictions = model(image_tensor)

    # Extract predictions
    pred_boxes = predictions[0]['boxes']
    pred_labels = predictions[0]['labels']
    pred_scores = predictions[0]['scores']
    pred_masks = predictions[0]['masks']

    # Threshold to filter out low-confidence detections
    threshold = 0.5
    keep = pred_scores > threshold

    # Convert to numpy array for further processing
    image_np = np.array(image)

    # Loop over detections
    for i in range(len(pred_boxes)):
        if keep[i]:
            # Bounding Box Method
            box = pred_boxes[i].cpu().numpy()
            x1, y1, x2, y2 = box

            # Calculate center coordinates from bounding box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            print(f"Bounding Box Center Coordinates (x, y): ({center_x}, {center_y})")

            # Mask Method
            mask = pred_masks[i, 0].cpu().numpy()
            mask_binary = (mask > 0.5).astype(np.uint8)

            # Find contours
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Use the largest contour
                contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(contour)

                if M["m00"] != 0:
                    # Calculate center of mass
                    center_x_mask = int(M["m10"] / M["m00"])
                    center_y_mask = int(M["m01"] / M["m00"])
                    print(f"Mask Center Coordinates (x, y): ({center_x_mask}, {center_y_mask})")

                    # Draw contours and center
                    cv2.drawContours(image_np, [contour], -1, (0, 255, 0), 2)
                    cv2.circle(image_np, (center_x_mask, center_y_mask), 5, (255, 0, 0), -1)

            # Draw bounding boxes
            cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            # Convert image coordinates to real-world coordinates
            u = center_x  # or center_x_mask if using mask method
            v = center_y  # or center_y_mask if using mask method

            X = (u - cx) * depth / fx
            Y = (v - cy) * depth / fy
            Z = depth

            print(f"Real-world Coordinates (X, Y, Z): ({X:.2f}, {Y:.2f}, {Z:.2f})")

    # Display the frame
    cv2.imshow('Webcam', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()