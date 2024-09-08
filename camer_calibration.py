import cv2
import numpy as np
import glob


Ch_Dim = (8, 6)
Sq_size = 9
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

obj_3D = np.zeros((Ch_Dim[0] * Ch_Dim[1], 3), np.float32)
index = 0
for i in range(Ch_Dim[0]):
    for j in range(Ch_Dim[1]):
        obj_3D[index][0] =  i * Sq_size
        obj_3D[index][1] = j * Sq_size
        index += 1
obj_points_3D = []
img_points_2D = []

image_files = glob.glob(r'C:\Users\shadr\PycharmProjects\TestAssignment\CalibrationPhotos\*.jpg')


for image in image_files:

    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(img, Ch_Dim, None)
    if ret:
        obj_points_3D.append(obj_3D)
        corners2 = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)
        img_points_2D.append(corners2)

        img = cv2.drawChessboardCorners(img, Ch_Dim, corners2, ret)

ret, mtx, dist, R_vecs, T_vecs = cv2.calibrateCamera(obj_points_3D, img_points_2D, gray.shape[::-1],
                                                     None, None)
print(mtx, dist)
print("calibrated")
calib_data_path = r'C:\Users\shadr\PycharmProjects\TestAssignment'
np.savez(
    f"{calib_data_path}/CalibrationMatrix_college_cpt",
    CameraMatrix=mtx,
    Distortion=dist,
    RotationalV=R_vecs,
    TranslationV=T_vecs
)