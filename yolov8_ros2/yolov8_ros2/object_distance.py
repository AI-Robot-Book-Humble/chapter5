import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from message_filters import Subscriber, ApproximateTimeSynchronizer
from std_msgs.msg import String

import cv2
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import ros2_numpy

from ultralytics import YOLO


class ObjectDistance(Node):

    def __init__(self, **args):
        super().__init__('object_distance')

        self.segmentation_model = YOLO("yolov8m-seg.pt")

        self.bridge = CvBridge()

        self.sub_info = Subscriber(
            self, CameraInfo, '/camera/camera/aligned_depth_to_color/camera_info')
        self.sub_color = Subscriber(
            self, Image, '/camera/camera/color/image_raw')
        self.sub_depth = Subscriber(
            self, Image, '/camera/camera/aligned_depth_to_color/image_raw')
        self.ts = ApproximateTimeSynchronizer(
            [self.sub_info, self.sub_color, self.sub_depth], 10, 0.1)
        self.ts.registerCallback(self.images_callback)
        
        self.pub_distance = self.create_publisher(String, 'object_distance', 10)

    def images_callback(self, msg_info, msg_color, msg_depth):
        try:
            img_color = CvBridge().imgmsg_to_cv2(msg_color, 'bgr8')
            img_depth = CvBridge().imgmsg_to_cv2(msg_depth, 'passthrough')
        except CvBridgeError as e:
            self.get_logger().warn(str(e))
            return

        if img_color.shape[0:2] != img_depth.shape[0:2]:
            self.get_logger().warn('カラーと深度の画像サイズが異なる')
            return
        
        image = ros2_numpy.numpify(msg_color)
        depth = ros2_numpy.numpify(msg_depth)
        segmentation_result = self.segmentation_model(image)
        annotated_frame = segmentation_result[0].plot()

        for index, cls in enumerate(segmentation_result[0].boxes.cls):
            class_index = int(cls.cpu().numpy())
            name = segmentation_result[0].names[class_index]
            mask = segmentation_result[0].masks.data.cpu().numpy()[index, :, :].astype(int)
            obj = depth[mask == 1]
            obj = obj[~np.isnan(obj)]
            avg_distance = np.mean(obj) if len(obj) else np.inf
            print(avg_distance)

        #self.pub_distance.publish(String(data=str(all_objects)))

        cv2.imshow('result', annotated_frame)
        cv2.waitKey(1)


def main():
    rclpy.init()
    object_distance = ObjectDistance()
    try:
        rclpy.spin(object_distance)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()
