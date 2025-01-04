import numpy as np
import json
import cv2
import os

path_left = "../static/data/left/"
path_right = "../static/data/right/"
output_name = "parameters_test.json"  # marking it as test is recommended to avoid unwanted overwriting
camera_width = 1280
camera_height = 480

# define chessboard parameter
CHESSBOARD_SIZE = (6,
                   9)  # not the amount of boxes but the amount of chessboard corner points! e.g., a square made by 9 small squares has chessboard size of (2, 2)
CHESSBOARD_SQUARE_SIZE = 25  # mm
img_list_left = sorted(os.listdir(path_left))
img_list_right = sorted(os.listdir(path_right))

objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * CHESSBOARD_SQUARE_SIZE

# define termination rule for detecting corner points
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# declare lists for saving corner points
obj_points = []
img_points_left = []
img_points_right = []

# read images and detect corner points
for i in range(len(img_list_left)):
    img_l = cv2.imread(path_left + img_list_left[i])
    print(img_list_left[i])
    img_r = cv2.imread(path_right + img_list_right[i])
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    # detect corner points
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHESSBOARD_SIZE, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHESSBOARD_SIZE, None)

    # if corner point is detected in both left and right images, store it to the list
    if ret_l and ret_r:
        obj_points.append(objp)

        # Obtain more accurate corner coordinates through subpixel-level corner detection.
        cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        img_points_left.append(corners_l)
        cv2.drawChessboardCorners(img_l, CHESSBOARD_SIZE, corners_l, ret_l)
        cv2.imshow("Chessboard Corners - Left", cv2.resize(img_l, (img_l.shape[1] // 2, img_l.shape[0] // 2)))
        cv2.waitKey(50)

        cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
        img_points_right.append(corners_r)
        cv2.drawChessboardCorners(img_r, CHESSBOARD_SIZE, corners_r, ret_r)
        cv2.imshow("Chessboard Corners - Right", cv2.resize(img_r, (img_r.shape[1] // 2, img_r.shape[0] // 2)))
        cv2.waitKey(50)

cv2.destroyAllWindows()

print("Number of chessboard images used for calibration: ", len(obj_points))

# calibrate camera
ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(obj_points, img_points_left, gray_l.shape[::-1], None,
                                                             None)
ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(obj_points, img_points_right, gray_r.shape[::-1], None,
                                                             None)

# calibrate stereo camera
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
ret, K1, d1, K2, d2, R, T, E, F = cv2.stereoCalibrate(
    obj_points, img_points_left, img_points_right,
    mtx_l, dist_l, mtx_r, dist_r,
    gray_l.shape[::-1], criteria=criteria, flags=flags)

print("left_intrinsic_matrix K1：\n", K1)
print("left_distortion d1：\n", d1)
print("right_intrinsic_matrix K2：\n", K2)
print("right_distortion d2：\n", d2)
print("rotation_matrix R：\n", R)
print("translation_matrix T：\n", T)
print("* essential matrix E：\n", E)
print("* fundamental matrix F：\n", F)

# store as dictionary
calibration_results = {
    "camera_width": camera_width,
    "camera_height": camera_height,
    "left_intrinsic_matrix": K1.tolist(),  # from ndarray to list
    "right_intrinsic_matrix": K2.tolist(),
    "left_distortion": d1.flatten().tolist(),  # from 2D ndarray to list
    "right_distortion": d2.flatten().tolist(),
    "rotation_matrix": R.tolist(),
    "translation_matrix": T.flatten().tolist(),
}

print("Result Dictionary：")
print(json.dumps(calibration_results, indent=4))

# save dictionary as json file
with open('../parameters_test.json', 'w') as f:
    json.dump(calibration_results, f, indent=4)

print("calibration parameters have been saved as ./parameters_test.json")
