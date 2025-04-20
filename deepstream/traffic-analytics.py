import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class TrafficAnalyticsSubscriber(Node):

    def __init__(self):
        super().__init__('traffic_analytics_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'Received detection data: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    traffic_analytics_subscriber = TrafficAnalyticsSubscriber()
    try:
        rclpy.spin(traffic_analytics_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        traffic_analytics_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()