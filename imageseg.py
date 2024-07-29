import os
import cv2
from ultralytics import SAM

# Load a model
model = SAM("sam_b.pt")

# Display model information (optional)
model.info()

# Run inference
results = model('/home/adithyadk/Desktop/model-ai/dataset/dataset4/010101.v1i.yolov9/train/images/WhatsApp-Image-2024-07-29-at-9-52-26-AM_jpeg.rf.b37802419dc85b7882643a1b2e245c25.jpg')

# Save segmented images in a different directory
output_dir = '/home/adithyadk/Desktop/model-ai/output_images'
os.makedirs(output_dir, exist_ok=True)

for i, result in enumerate(results):
    # Save the segmented image
    output_image_path = os.path.join(output_dir, f"segmented_image_{i}.jpg")
    cv2.imwrite(output_image_path, result.masks.pred)

    # Display the segmented image
    cv2.imshow(f"Segmented Image {i}", result.masks.pred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()