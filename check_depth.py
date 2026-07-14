
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

class DepthCheck(Node):
    def __init__(self):
        super().__init__('depth_check')
        self.bridge = CvBridge()
        self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.cb, 10)
        self.vals = []
    def cb(self, msg):
        d = self.bridge.imgmsg_to_cv2(msg)
        h, w = d.shape
        roi = d[h//2-10:h//2+10, w//2-10:w//2+10].astype(float)
        roi = roi[roi > 0]  # 0 = 무효 픽셀 제외
        if roi.size == 0:
            print('중앙 ROI 전체 무효 — 벽이 너무 가깝거나 반사 문제')
            return
        m = np.median(roi) / 1000.0  # mm -> m
        self.vals.append(m)
        print(f'중앙 depth median = {m:.3f} m  (유효픽셀 {roi.size}/400)')
        if len(self.vals) >= 10:
            arr = np.array(self.vals)
            print(f'--- 10프레임 요약: mean={arr.mean():.3f}m  std={arr.std()*1000:.1f}mm ---')
            rclpy.shutdown()

rclpy.init()
rclpy.spin(DepthCheck())