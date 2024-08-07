import rclpy
from std_msgs.msg import Float32MultiArray
#import geometry_msgs
from rclpy.node import Node

class ValueSubscriber(Node):
    def __init__(self):
        super().__init__('value_subscriber')
        self.subscription= self.create_subscription(Float32MultiArray)
        self.x = 0.0
        self.y = 0.0 
        self.z = 0.0
        self.height = 0.0
        self.width = 0.0
    def value_callback(self,msg):
        if len(msg.data) == 5:
            self.x=msg.data[0]
            self.y=msg.data[1]
            self.z=msg.data[2]
            self.height = msg.data[3]
            self.width = msg.data[4]
            self.get_logger().info(f"Recieved Values :X={self.f}, Y={self.y}, Z={self.z}, Height={self.height}, Width={self.width}")
        else:
            self.get_logger().warn("Recieved message with incorrect number of values")
def main(args=None):
    rclpy.init(args=args)
    value_subscriber = ValueSubscriber()
    rclpy.spin(value_subscriber)
    value_subscriber.destroy_node()
    rclpy.shutdown()      
    

if __name__ == '__main__':
    main()