import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(Image, 'detected_frames', 10)
        self.timer = self.create_timer(1.0 / 15, self.timer_callback)  # 15 FPS
        self.cap = cv2.VideoCapture(0)  # Use the system's main camera
        self.bridge = CvBridge()

        # Load YOLOv8 model
        self.model = YOLO(r"/home/adithyadk/runs/segment/train6/weights/best(seg).pt")

        # Camera intrinsic parameters
        self.fx = 10447.820616852037  # Focal length in x (pixels)
        self.fy = 10477.457878200092  # Focal length in y (pixels)
        self.cx = 305.83263140233214  # Principal point x (pixels)
        self.cy = 954.334079526126  # Principal point y (pixels)
        self.depth = 0.45  # Depth of the object from the camera (in meters)

        # Create camera matrix
        self.camera_matrix = np.array([[self.fx, 0, self.cx],
                                       [0, self.fy, self.cy],
                                       [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros(5)  # Assuming no lens distortion for simplicity

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            # Undistort the frame using calibration parameters
            frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
            
            # Process frame with YOLOv8 model
            results = self.model(frame)
            if results:  # Check if any objects are detected
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = self.model.names[int(box.cls)]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
                self.publisher_.publish(msg)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    camera_publisher = CameraPublisher()
    rclpy.spin(camera_publisher)
    camera_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

#"/home/adithyadk/runs/segment/train6/weights/best(seg).pt"