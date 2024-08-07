import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class ImageSubscriber(Node):
    def __init__(self, output_file):
        super().__init__('image_subscriber')
        self.output_file = output_file
        self.subscription = self.create_subscription(
            Image, 'camera/image_raw', self.listener_callback, 10)
        self.br = CvBridge()

    def listener_callback(self, msg):
        self.get_logger().info('Receiving image frame')
        current_frame = self.br.imgmsg_to_cv2(msg, 'bgr8')
        if current_frame is None:
            return

        # Process the image to extract coordinates
        # For example, let's assume we want to find the center of the image
        height, width, _ = current_frame.shape
        center_x = width // 2
        center_y = height // 2

        # Save the coordinates to the text file
        with open(self.output_file, 'a') as file:
            file.write(f"{center_x},{center_y},{height},{width}\n")

def main(args=None):
    rclpy.init(args=args)
    file_name="coordinates.txt"
    output_file = ("/home/adithyadk/testing",file_name)  # Specify the output file location
    image_subscriber = ImageSubscriber(output_file)
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



    
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        cv2.namedWindow("camera", cv2.WINDOW_NORMAL)  # Create the OpenCV window
        self.subscription = self.create_subscription(
            Image, 'camera/image_raw', self.listener_callback, 10)
        self.br = CvBridge()

    def listener_callback(self, msg):
        self.get_logger().info('Receiving image frame')
        current_frame = self.br.imgmsg_to_cv2(msg, 'bgr8')
        if current_frame is None:
            cv2.imshow("camera", current_frame)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
