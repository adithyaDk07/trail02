import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import cv2
from ultralytics import YOLO

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'object_data', 10)
        self.timer = self.create_timer(1/15, self.timer_callback)  # 15 FPS

        # Load YOLO model
        self.model = YOLO(r"/home/adithyadk/runs/segment/train6/weights/best(seg).pt")
        
        self.cap = cv2.VideoCapture(0)  # Using the system's main camera

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Failed to capture image")
            return
        
        # Perform object detection
        detections = self.model(frame)
        for detection in detections.xyxy[0]:
            x1, y1, x2, y2, conf, cls = map(int, detection)
            width = x2 - x1
            height = y2 - y1
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Placeholder for real-world coordinate transformation
            real_world_coords = self.calculate_real_world_coordinates(center_x, center_y, width, height)
            
            data = Float32MultiArray()
            data.data = [real_world_coords[0], real_world_coords[1], real_world_coords[2], width, height]
            self.publisher_.publish(data)
            self.get_logger().info(f"Published object data: {data.data}")

    def calculate_real_world_coordinates(self, x, y, width, height):
        # Placeholder function for transforming pixel coordinates to real-world coordinates
        # Assuming a simple pinhole camera model for this example
        fx, fy = 10447.820616852037, 10477.457878200092  # Focal lengths
        cx, cy = 10477.457878200092, 10477.457878200092  # Principal point (center of the image)
        Z = 0.45  # Assume depth is 1 meter for simplicity

        X = (x - cx) * Z / fx
        Y = (y - cy) * Z / fy
        return (X, Y, Z)

def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




