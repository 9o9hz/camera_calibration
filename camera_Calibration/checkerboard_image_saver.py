import cv2
import datetime
import os

save_dir = "/home/j/check/capture"
os.makedirs(save_dir, exist_ok=True)

# cap = cv2.VideoCapture('/dev/video2')
cap = cv2.VideoCapture('/dev/video2', cv2.CAP_V4L2)

window_name = "Video"
latest_frame = None

def mouse_callback(event, x, y, flags, param):
    global latest_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        if latest_frame is not None:
            filename = datetime.datetime.now().strftime(
                os.path.join(save_dir, "capture_%Y%m%d_%H%M%S.png")
            )
            ok = cv2.imwrite(filename, latest_frame)
            if ok:
                print(f"{filename} 이미지가 저장되었습니다.")
            else:
                print("이미지 저장 실패")

cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라에서 프레임을 가져올 수 없습니다.")
        break

    latest_frame = frame.copy()
    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()