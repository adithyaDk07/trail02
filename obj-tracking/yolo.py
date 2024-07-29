import cv2
import numpy as np

class YOLO:
    def __init__(self, weights_path, config_path, classes_path):
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self.classes = self.load_classes(classes_path)

    def load_classes(self, classes_path):
        with open(classes_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes

    def preprocess_image(self, image):
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        return blob
    
    def detect_objects(self, image):
        blob = self.preprocess_image(image)
        self.net.setInput(blob)
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        outputs = self.net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x, center_y, width, height = detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        return boxes, confidences, class_ids