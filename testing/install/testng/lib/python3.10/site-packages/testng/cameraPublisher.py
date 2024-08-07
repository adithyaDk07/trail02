import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import cv2
import torch
from ultralytics import YOLO

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'object_data', 10)
        self.timer = self.create_timer(2.0, self.timer_callback)  # 1 coordinate per 5 seconds

        # Load YOLO model
        self.model = YOLO(r"/home/adithyadk/runs/segment/train6/weights/best(seg).pt")
        
        self.cap = cv2.VideoCapture(0)  # Using the system's main camera

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Failed to capture image")
            return
        
        # Perform object detection
        results = self.model(frame)
        detections = results[0].boxes  # Access the boxes attribute directly
        
        if len(detections) > 0:  # Check if any detections are present
            detection = detections[0]  # Use the first detection for simplicity
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            width = x2 - x1
            height = y2 - y1
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Placeholder for real-world coordinate transformation
            real_world_coords = self.calculate_real_world_coordinates(center_x, center_y, width, height)
            
            data = Float32MultiArray()
            data.data = [round(real_world_coords[0], 4), round(real_world_coords[1], 4), round(real_world_coords[2], 4), round(width / 1000.0, 4), round(height / 1000.0, 4)]  # Convert width and height to meters, round to 4 decimal places
            self.publisher_.publish(data)
            self.get_logger().info(f"Published object data: {data.data}")
        else:
            # Publish a special message indicating no detections
            no_detection_data = Float32MultiArray()
            no_detection_data.data = [0, 0, 0, 0, 0]
            self.publisher_.publish(no_detection_data)
            self.get_logger().info("No detections")

    def calculate_real_world_coordinates(self, x, y, width, height):
        # Placeholder function for transforming pixel coordinates to real-world coordinates
        # Assuming a simple pinhole camera model for this example
        fx, fy = 10447.82, 10477.45  # Focal lengths
        cx, cy = 10477.45, 10477.45  # Principal point (center of the image)
        Z = 0.45  # Assume depth is 1 meter for simplicity

        X = (x - cx) * Z / fx
        Y = (y - cy) * Z / fy
        return X, Y, Z

def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()