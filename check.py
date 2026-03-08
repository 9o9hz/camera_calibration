#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from rclpy.qos import qos_profile_sensor_data

import cv2
import numpy as np
import argparse
import math
import os


class CameraMeasureROS2Node(Node):
    def __init__(self, args):
        super().__init__('camera_measure_ros2_node')
        self.args = args

        self.points = []
        self.hover_pt = None
        self.last_measure_info = None
        self.latest_frame = None

        # Start with default values, will be updated by calibration
        self.m_per_pixel_x = 0.0
        self.m_per_pixel_y = 0.0
        self.grid_interval_m = args.grid_interval_m

        self.grid_px_x = 1
        self.grid_px_y = 1
        # self.update_grid_px() # Will be called after calibration

        if 'compressed' in args.topic:
            self.msg_type = CompressedImage
        else:
            self.msg_type = Image

        self.subscription = self.create_subscription(
            self.msg_type,
            args.topic,
            self.image_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info(f"Subscribing to: {args.topic}")

        self.window_name = "Camera Measure Tool (ROS2)"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\n[사용 방법 - 미터 퍼 픽셀 (m/px) 캘리브레이션 (간소화)]")
        print("프레임의 왼쪽 상단이 원점(0m, 0m)으로 간주됩니다.")
        print("바닥에 표시된 0.5m 간격의 격자점을 순서대로 클릭해주세요.")
        print("1. [클릭 1] X축 방향으로 0.5m 떨어진 점 (0.5m, 0m)을 클릭하세요.")
        print("2. [클릭 2] Y축 방향으로 0.5m 떨어진 점 (0m, 0.5m)을 클릭하세요.")
        print("-" * 50)
        print("캘리브레이션 완료 후에는 일반 측정 모드가 활성화됩니다.")
        print("  - 좌클릭: 점 선택 / 우클릭: 마지막 점 취소 / 가운데클릭: 전체 점 삭제")
        print("  - 's': 저장 / 'r': 리셋 / 'u': 마지막 점 취소 / 'q': 종료\n")

    def update_grid_px(self):
        if self.m_per_pixel_x > 0:
            self.grid_px_x = max(1, int(self.grid_interval_m / self.m_per_pixel_x))
        if self.m_per_pixel_y > 0:
            self.grid_px_y = max(1, int(self.grid_interval_m / self.m_per_pixel_y))

    def mouse_callback(self, event, x, y, flags, param):
        self.hover_pt = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 2: # Limit calibration clicks to 2
                self.points.append((x, y))
                print(f"[CALIB CLICK {len(self.points)}] u={x}, v={y}")
            else: # Measurement mode
                self.points.append((x,y))
                print(f"[MEASURE CLICK {len(self.points) - 2}] u={x}, v={y}")


            # Calibration phase
            if len(self.points) == 1:
                p_xaxis = self.points[0]
                px_x = p_xaxis[0]

                if px_x > 0:
                    self.m_per_pixel_x = self.grid_interval_m / px_x
                    print(f"  - X-axis pixel: {px_x} px")
                    print(f"  - [X축 m/px 계산] {self.grid_interval_m:.3f}m / {px_x}px = {self.m_per_pixel_x:.9f} m/px")
                    self.update_grid_px()
                else:
                    print("  - [경고] X축 픽셀 위치가 0입니다. 다른 점을 선택하세요.")
                print("-> X축 점 선택됨. 다음, Y축 점을 클릭하세요.")

            elif len(self.points) == 2:
                p_yaxis = self.points[1]
                px_y = p_yaxis[1]

                if px_y > 0:
                    self.m_per_pixel_y = self.grid_interval_m / px_y
                    print(f"  - Y-axis pixel: {px_y} px")
                    print(f"  - [Y축 m/px 계산] {self.grid_interval_m:.3f}m / {px_y}px = {self.m_per_pixel_y:.9f} m/px")
                    self.update_grid_px()
                else:
                    print("  - [경고] Y축 픽셀 위치가 0입니다. 다른 점을 선택하세요.")

                print("-" * 50)
                print("[캘리브레이션 완료] 이제부터 일반 측정 모드로 작동합니다.")
                print("-" * 50)

            # Measurement phase (after calibration)
            elif len(self.points) > 2:
                # Need at least two measurement points (total 4 points)
                if len(self.points) < 4:
                    return

                p1 = self.points[-2]
                p2 = self.points[-1]

                du = abs(p2[0] - p1[0])
                dv = abs(p2[1] - p1[1])
                dist_px = math.sqrt(du**2 + dv**2)
                dist_m_x = du * self.m_per_pixel_x
                dist_m_y = dv * self.m_per_pixel_y
                dist_m = math.sqrt(dist_m_x**2 + dist_m_y**2)

                info = {
                    "p1": p1, "p2": p2, "du": du, "dv": dv,
                    "dist_px": dist_px, "dist_m": dist_m,
                    "dist_m_x": dist_m_x, "dist_m_y": dist_m_y
                }
                self.last_measure_info = info

                print(f"  - p1: {p1}, p2: {p2}")
                print(f"  - Δu: {du} px, Δv: {dv} px")
                print(f"  - Pixel Distance: {dist_px:.2f} px")
                print(f"  - Est. Distance (X): {dist_m_x:.3f} m")
                print(f"  - Est. Distance (Y): {dist_m_y:.3f} m")
                print(f"  - Est. Total Distance: {dist_m:.3f} m")
                print("-" * 50)


        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                removed = self.points.pop()
                print(f"[UNDO] removed point {removed}")
                if len(self.points) < 4:
                    self.last_measure_info = None

        elif event == cv2.EVENT_MBUTTONDOWN:
            self.points.clear()
            self.m_per_pixel_x = 0.0
            self.m_per_pixel_y = 0.0
            self.last_measure_info = None
            print("[CLEAR] all points cleared. Restarting calibration.")
            print("1. [클릭 1] X축 방향으로 0.5m 떨어진 점 (0.5m, 0m)을 클릭하세요.")

    def decode_image(self, msg):
        np_arr = np.frombuffer(msg.data, dtype=np.uint8)

        if self.msg_type == Image:
            # This part handles different image encodings from ROS
            if msg.encoding == 'yuyv':
                frame = cv2.cvtColor(np_arr.reshape((msg.height, msg.width, 2)), cv2.COLOR_YUV2BGR_YUYV)
            elif msg.encoding == 'rgb8':
                frame = cv2.cvtColor(np_arr.reshape((msg.height, msg.width, 3)), cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'bgr8':
                frame = np_arr.reshape((msg.height, msg.width, 3))
            else: # Fallback for compressed or other types
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else: # CompressedImage
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        return frame

    def draw_grid(self, img):
        out = img.copy()
        h, w = out.shape[:2]

        if self.grid_px_x > 1:
            for x in range(0, w, self.grid_px_x):
                cv2.line(out, (x, 0), (x, h - 1), (200, 200, 200), 1)

        if self.grid_px_y > 1:
            for y in range(0, h, self.grid_px_y):
                cv2.line(out, (0, y), (w - 1, y), (200, 200, 200), 1)

        return out

    def draw_info(self, img):
        out = img.copy()
        origin = (0, 0)

        if self.hover_pt is not None:
            x, y = self.hover_pt
            cv2.circle(out, (x, y), 4, (0, 255, 255), -1)
            cv2.putText(
                out, f"u={x}, v={y}", (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
            )

        # Draw calibration points and lines
        if len(self.points) >= 1:
            p_xaxis = self.points[0]
            cv2.circle(out, p_xaxis, 6, (255, 0, 100), -1)
            cv2.line(out, origin, p_xaxis, (255, 100, 0), 2)
            cv2.putText(out, f"C1(X): {p_xaxis}", (p_xaxis[0] + 8, p_xaxis[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 100), 2)

        if len(self.points) >= 2:
            p_yaxis = self.points[1]
            cv2.circle(out, p_yaxis, 6, (100, 0, 255), -1)
            cv2.line(out, origin, p_yaxis, (100, 0, 255), 2)
            cv2.putText(out, f"C2(Y): {p_yaxis}", (p_yaxis[0] + 8, p_yaxis[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 0, 255), 2)

        # Draw measurement points
        measure_points = self.points[2:]
        for i, (x, y) in enumerate(measure_points):
            cv2.circle(out, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(out, f"P{i+1}({x},{y})", (x + 8, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if len(measure_points) >= 2:
            for i in range(1, len(measure_points)):
                cv2.line(out, measure_points[i - 1], measure_points[i], (0, 255, 0), 2)


        y0 = 25
        # Main instructions / status text
        infos = []
        if len(self.points) == 0:
            infos.append("CALIBRATION: Click X-axis point (0.5m, 0)")
        elif len(self.points) == 1:
            infos.append("CALIBRATION: Click Y-axis point (0, 0.5m)")
        else:
            infos.append("MEASURE MODE: Click points to measure distance")

        # General info
        infos.extend([
            f"m/px X: {self.m_per_pixel_x:.7f}" if self.m_per_pixel_x else "m/px X: Not calibrated",
            f"m/px Y: {self.m_per_pixel_y:.7f}" if self.m_per_pixel_y else "m/px Y: Not calibrated",
            f"Grid: {self.grid_interval_m:.2f} m ({self.grid_px_x}px, {self.grid_px_y}px)",
            "s: save, r: reset, u: undo, q: quit"
        ])


        if self.last_measure_info is not None:
            infos.append(f"Last Est. Dist = {self.last_measure_info['dist_m']:.3f} m")

        for i, text in enumerate(infos):
            cv2.putText(
                out, text, (10, y0 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
            )

        return out

    def save_points(self):
        try:
            calib_points = np.array(self.points[:2], dtype=np.int32)
            measure_points = np.array(self.points[2:], dtype=np.int32)

            np.savez(
                self.args.out_npz,
                calibration_points=calib_points,
                measurement_points=measure_points,
                m_per_pixel_x=self.m_per_pixel_x,
                m_per_pixel_y=self.m_per_pixel_y,
                grid_interval_m=self.grid_interval_m
            )

            with open(self.args.out_txt, 'w') as f:
                f.write("# Camera Measure Tool Clicked Points (ROS2)\n")
                f.write(f"# topic: {self.args.topic}\n")
                f.write("# Calibrated Values\n")
                f.write(f"m_per_pixel_x: {self.m_per_pixel_x:.9f}\n")
                f.write(f"m_per_pixel_y: {self.m_per_pixel_y:.9f}\n")
                f.write(f"grid_interval_m: {self.grid_interval_m}\n\n")

                f.write("# Calibration Points (C1:X-axis, C2:Y-axis)\n")
                if len(calib_points) > 0:
                    f.write(f"C1(X): {calib_points[0][0]}, {calib_points[0][1]}\n")
                if len(calib_points) > 1:
                    f.write(f"C2(Y): {calib_points[1][0]}, {calib_points[1][1]}\n")

                f.write("\n# Measurement Points\n")
                if len(measure_points) == 0:
                    f.write("# No measurement points.\n")
                else:
                    for i, (x, y) in enumerate(measure_points):
                        f.write(f"P{i+1}: {x}, {y}\n")

            print(f"[INFO] 저장 완료:")
            print(f"       NPZ -> {os.path.abspath(self.args.out_npz)}")
            print(f"       TXT -> {os.path.abspath(self.args.out_txt)}")

        except Exception as e:
            print(f"[ERROR] 저장 중 오류 발생: {e}")

    def process_key(self, key):
        if key == ord('q'):
            rclpy.shutdown()
            return

        elif key == ord('r'):
            self.points.clear()
            self.m_per_pixel_x = 0.0
            self.m_per_pixel_y = 0.0
            self.last_measure_info = None
            print("[RESET] all points cleared. Restarting calibration.")
            print("1. [클릭 1] X축 방향으로 0.5m 떨어진 점 (0.5m, 0m)을 클릭하세요.")

        elif key == ord('u'):
            if self.points:
                removed = self.points.pop()
                print(f"[UNDO] removed point {removed}")
                if len(self.points) < 4:
                    self.last_measure_info = None

        elif key == ord('s'):
            self.save_points()

    def image_callback(self, msg):
        try:
            frame = self.decode_image(msg)
            if frame is None:
                return

            self.latest_frame = frame

            view = self.draw_grid(frame)
            view = self.draw_info(view)

            cv2.imshow(self.window_name, view)
            key = cv2.waitKey(1) & 0xFF
            self.process_key(key)

        except Exception as e:
            self.get_logger().error(f"Image callback error: {e}")


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', type=str, default='/image_raw/compressed',
                        help='구독할 ROS2 이미지 토픽')
    parser.add_argument('--out-npz', type=str, default='camera_measure_points.npz',
                        help='저장할 측정 npz 파일명')
    parser.add_argument('--out-txt', type=str, default='camera_measure_points.txt',
                        help='저장할 측정 txt 파일명')
    parser.add_argument('--m-per-pixel-x', type=float, default=0.003578125,
                        help='x축 meter per pixel')
    parser.add_argument('--m-per-pixel-y', type=float, default=0.0025,
                        help='y축 meter per pixel')
    parser.add_argument('--grid-interval-m', type=float, default=0.5,
                        help='그리드 간격(m)')
    parsed_args = parser.parse_args()

    node = CameraMeasureROS2Node(parsed_args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()