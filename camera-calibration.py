import cv2
import numpy as np
import imutils
import requests

# Camera calibration parameters
chessboard_size = (9, 6)
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

# Replace the below URL with your own. Make sure to add "/shot.jpg" at the end.
url = "http://192.168.29.210:8080/shot.jpg"

# Calibrate the camera
cap = cv2.VideoCapture(url)
while True:
    ret, img = cap.read()

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray_img, chessboard_size, None)

    if ret:
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        objpoints.append(objp)
        imgpoints.append(corners)

    cv2.imshow('Chessboard Corners', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_img.shape[::-1], None, None)

print("Camera Matrix (Intrinsic Parameters):")
print(mtx)
print("\nDistortion Coefficients:")
print(dist)
print("\nRotation Vectors:")
print(rvecs)
print("\nTranslation Vectors:")
print(tvecs)

# Load the tracking model (replace 'your_model.xml' with the actual file path)
tracker = cv2.TrackerCSRT_create()
success, frame = cap.read()
bbox = cv2.selectROI("Tracking", frame, False)
tracker.init(frame, bbox)

while True:
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    img = imutils.resize(img, width=1000, height=1800)

    success, bbox = tracker.update(img)

    if success:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(img, p1, p2, (255, 0, 0), 2, 1)

    cv2.imshow("Android_cam", img)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()