import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge, CvBridgeError

from ultralytics import YOLO


class ObjectSegmentation(Node):

    def __init__(self, **args):
        super().__init__('object_segmentation')

        self.segmentation_model = YOLO("yolov8m-seg.pt")

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            'image_raw',
            self.image_callback,
            qos_profile_sensor_data)

    def image_callback(self, msg):
        try:
            img0 = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().warn(str(e))
            return

        segmentation_result = self.segmentation_model(img0)
        annotated_frame = segmentation_result[0].plot()

        cv2.imshow('result', annotated_frame)
        cv2.waitKey(1)


def main():
    rclpy.init()
    object_segmentation = ObjectSegmentation()
    try:
        rclpy.spin(object_segmentation)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()
