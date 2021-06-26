import numpy as np
import cv2 as cv
from pathlib import Path
# import glob
# import os
# NOTE: using pathlib.Path.rglob from the the pathlib module instead of using glob() from os


# FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS

# chessboard = (row, col)
chessboardSize = (9,6)

# image dimensions (640 × 480- for default opencv checkerboard image set)
frameSize = (640,480)

# accessing calibration images folder
imgPath = Path('/Users/yasmeen/Desktop/side_project_cabin/ContinuumRoboticsLab/calibimgs').rglob('*.jpg')


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)


# Arrays to store object points and image points from all the images.
objpoints = [] # 3D point in real world space
imgpoints = [] # 2D points in image plane.


for path in imgPath:

    img = cv.imread(str(path))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)
        # Refines the corner locations
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('chessboard calibration', img)
        cv.waitKey(1000)


# closes the image window after completion
cv.destroyAllWindows()



# CALIBRATION

# camera matrix, distortion coefficients, rotation and translation vectors
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

# Outputs Camera Matrix and Distortion Coefficients
print('=Camera Matrix=\n', cameraMatrix, '\n')
print('=Distortion Matrix=\n', dist, '\n')


# UNDISTORTION PROCESS


# refine the camera matrix based on a free scaling parameter using cv.getOptimalNewCameraMatrix().
# If the scaling parameter alpha=0, it returns undistorted image with minimum unwanted pixels.
# So it may even remove some pixels at image corners.
# If alpha=1, all pixels are retained with some extra black images.
# This function also returns an image ROI which can be used to crop the result.

# img = cv.imread('cali5.png')
# img = cv.imread('Image__2018-10-05__10-38-27.png')


img = cv.imread('/Users/yasmeen/Desktop/side_project_cabin/ContinuumRoboticsLab/calibimgs/left01.jpg')
cv.imshow('left01.jpg', img)

h,  w = img.shape[:2]

# Returns the new camera intrinsic matrix based on the free scaling parameter.
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))



# UNDISTORTION TYPES

# METHOD 1 (Undistort)
# easiest method: call undistort function and use ROI obtained to crop the result.
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# cropping the image with the ROI
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

# saves the calibrated image to a specified file name
cv.imwrite('calibratedImg1.png', dst)


# METHOD 2 (Undistort with Remapping)

# find a mapping function from the distorted image to the undistorted image
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)

# remap the image
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibratedImg2.png', dst)

# NOTE:
# can store the camera matrix and distortion coefficients using write functions
# in NumPy (np.savez, np.savetxt etc) for future uses.


# REPROJECTION ERROR (good estimation of just how exact the found parameters are)
# re-projection error is to zero, the more accurate the parameters we found are

mean_error = 0

for i in range(len(objpoints)):

    # given the intrinsic, distortion, rotation and translation matrices, transform the object point to image point
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)

    # calculate the absolute norm between what we got with our transformation and the corner finding algorithm
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)

    # average error = arithmetical mean of the errors calculated for all the calibration images.
    mean_error += error

# output final re-projection error
print( "total error: {}".format(mean_error/len(objpoints)) )




"""

Camera Calibration Results:

=Camera Matrix=
 [[536.07343567   0.         342.37038375]
 [  0.         536.01635124 235.53685495]
 [  0.           0.           1.        ]] 

=Distortion Matrix=
 [[-0.26509012 -0.04674346  0.00183301 -0.00031472  0.25231495]] 

total error: 0.04095727712912311
"""

