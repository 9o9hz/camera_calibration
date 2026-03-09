import cv2
import numpy as np
import os
import glob
import pickle

def calibrate_camera():
    # 가로,세로: 10 x 7
    CHECKERBOARD = (10, 7)
    square_size = 25.0  # mm

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objpoints = []
    imgpoints = []

    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp[:, :2] *= square_size

    images = glob.glob('/home/j/check/capture/*.png')
    print(f"찾은 이미지 수: {len(images)}")

    if len(images) == 0:
        raise FileNotFoundError("capture 폴더에 png 이미지가 없습니다.")

    image_size = None
    success_count = 0

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"이미지 읽기 실패: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_FAST_CHECK +
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            imgpoints.append(corners2)
            success_count += 1

            img_draw = cv2.drawChessboardCorners(img.copy(), CHECKERBOARD, corners2, ret)
            cv2.imshow('img', img_draw)
            cv2.waitKey(300)
        else:
            print(f"코너 검출 실패: {fname}")

    cv2.destroyAllWindows()

    if success_count == 0:
        raise RuntimeError("체커보드 코너를 한 장도 검출하지 못했습니다. CHECKERBOARD 설정과 이미지를 확인하세요.")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    print("Camera matrix : \n", mtx)
    print("\ndist : \n", dist)
    print("\nrvecs : \n", rvecs)
    print("\ntvecs : \n", tvecs)

    calibration_data = {
        'camera_matrix': mtx,
        'dist_coeffs': dist,
        'rvecs': rvecs,
        'tvecs': tvecs
    }

    with open('camera_calibration.pkl', 'wb') as f:
        pickle.dump(calibration_data, f)

    return calibration_data


if __name__ == "__main__":
    print("Performing new camera calibration...")
    calibration_data = calibrate_camera()
    print("Calibration done.")