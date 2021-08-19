import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import os
import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
from typing import List, Callable


# HELPER METHODS FROM OTHER FILES
from test_functions import detectAruco


# NOTE: several tutorials were followed, credited, and adapted into this codebase, namely:
# https://docs.opencv.org/master/d3/d81/tutorial_contrib_root.html
# https://www.andreasjakl.com/understand-and-apply-stereo-rectification-for-depth-maps-part-2/
# https://medium.com/analytics-vidhya/camera-calibration-with-opencv-f324679c6eb7

# cameras 1 & 3 setup paths
img_path_cam1 = glob.glob('./calibimgs_cam1_NEW2/*')
img_path_cam3 = glob.glob('./calibimgs_cam3_NEW2/*')
#---------------------------------------------------------- Camera Calibration ----------------------------------------------------------#

print("======== PHASE 1: CAMERA CALIBRATION ========")

# -------------------- CAMERA 1 CALIBRATION--------------------#

print("======== PHASE 1.1: CAMERA 1 CALIBRATION ========")

# Define size of chessboard target.
chessboard_size = (9 ,6)

# Define arrays to save detected points
obj_points_cam1 = []  # 3D points in real world space
img_points_cam1 = []  # 3D points in image plane

# Prepare grid and points to display
# Defining the world coordinates for 3D points
objp_cam1 = np.zeros((np.prod(chessboard_size) ,3) ,dtype=np.float32)

objp_cam1[: ,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1 ,2)

# read images
calibration_paths_cam1 = glob.glob('./calibimgs_cam1_NEW2/*')

# Iterate over images to find intrinsic matrix
# NOTE: Tqdm is a Python library used to display smart progress bars that show the progress of Python code execution (helpful for calibration phase)
for image_path_cam1 in tqdm(calibration_paths_cam1):

    # Load image
    image_cam1 = cv.imread(image_path_cam1)
    gray_image_cam1 = cv.cvtColor(image_cam1, cv.COLOR_BGR2GRAY)
    # find chessboard corners
    ret_cam1 ,corners_cam1 = cv.findChessboardCorners(gray_image_cam1, chessboard_size, None)

    if ret_cam1 == True:
        print("Detecting chessboard in progress")
        print(image_path_cam1)
        # define criteria for subpixel accuracy
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # refine corner location (to subpixel accuracy) based on criteria.
        corners2 = cv.cornerSubPix(gray_image_cam1, corners_cam1, (5 ,5), (-1 ,-1), criteria)
        obj_points_cam1.append(objp_cam1)
        img_points_cam1.append(corners_cam1)

        # Draw and display the corners
        cv.drawChessboardCorners(image_cam1, chessboard_size, corners2, ret_cam1)
        cv.imshow('chessboard calibration', image_cam1)
        cv.waitKey(1000)

# Calibrate camera
ret_cam1, K_cam1, dist_cam1, rvecs_cam1, tvecs_cam1 = cv.calibrateCamera(obj_points_cam1, img_points_cam1 ,gray_image_cam1.shape[::-1], None, None)

# Save parameters into numpy file
np.save("./camera_calib_params_cam1/ret", ret_cam1)
np.save("./camera_calib_params_cam1/K", K_cam1)
np.save("./camera_calib_params_cam1/dist", dist_cam1)
np.save("./camera_calib_params_cam1/rvecs", rvecs_cam1)
np.save("./camera_calib_params_cam1/tvecs", tvecs_cam1)


# Outputs Camera Matrix and Distortion Coefficients
print('=Camera Matrix (K)=\n', K_cam1, '\n')
print('=Distortion Coefficients=\n', dist_cam1, '\n')
print('=Rotational Vectors=\n', rvecs_cam1, '\n')
print('=Translational Vectors=\n', tvecs_cam1, '\n')
print('=Ret Value=\n', ret_cam1, '\n')



img_cam1 = cv.imread('/Users/yasmeen/Desktop/side_project_cabin/ContinuumRoboticsLab/calibimgs_cam1_NEW2/cam1_1.png')
cv.imshow('cam1_1.png', img_cam1)

h,  w = img_cam1.shape[:2]

# Returns the new camera intrinsic matrix based on the free scaling parameter.
newCameraMatrix_cam1, roi_cam1 = cv.getOptimalNewCameraMatrix(K_cam1, dist_cam1, (w ,h), 1, (w ,h))



# UNDISTORTION
# undistort function and use ROI obtained to crop the result with OpenCV.
dst = cv.undistort(img_cam1, K_cam1, dist_cam1, None, newCameraMatrix_cam1)

# cropping the image with the ROI
x_cam1, y_cam1, w_cam1, h_cam1 = roi_cam1
dst = dst[y_cam1: y_cam1 +h_cam1, x_cam1: x_cam1 +w_cam1]
cv.imwrite('calibratedImg1.png', dst)

# Calculate projection error.
mean_error_cam1 = 0
for i in range(len(obj_points_cam1)):
    img_points2, _ = cv.projectPoints(obj_points_cam1[i] ,rvecs_cam1[i] ,tvecs_cam1[i], K_cam1, dist_cam1)
    error_cam1 = cv.norm(img_points_cam1[i], img_points2, cv.NORM_L2 ) /len(img_points2)
    mean_error_cam1 += error_cam1

# output final re-projection error
print( "total error: {}".format(mean_error_cam1 /len(obj_points_cam1)))




# -------------------- CAMERA 3 CALIBRATION--------------------#

print("======== PHASE 1.3: CAMERA 3 CALIBRATION ========")

# Define arrays to save detected points
obj_points_cam3 = []  # 3D points in real world space
img_points_cam3 = []  # 3D points in image plane

# Prepare grid and points to display

objp_cam3 = np.zeros((np.prod(chessboard_size) ,3) ,dtype=np.float32)


objp_cam3[: ,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1 ,2)

# read images

calibration_paths_cam3 = glob.glob('./calibimgs_cam3_NEW2/*')

# Iterate over images to find intrinsic matrix
for image_path_cam3 in tqdm(calibration_paths_cam3):

    # Load image
    image_cam3 = cv.imread(image_path_cam3)
    gray_image_cam3 = cv.cvtColor(image_cam3, cv.COLOR_BGR2GRAY)
    # find chessboard corners
    ret_cam3 ,corners_cam3 = cv.findChessboardCorners(gray_image_cam3, chessboard_size, None)

    if ret_cam3 == True:
        print("Detecting chessboard in progress")
        print(image_path_cam3)
        # define criteria for subpixel accuracy
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # refine corner location (to subpixel accuracy) based on criteria.
        corners2_cam3 = cv.cornerSubPix(gray_image_cam3, corners_cam3, (5 ,5), (-1 ,-1), criteria)
        obj_points_cam3.append(objp_cam3)
        img_points_cam3.append(corners_cam3)

        # Draw and display the corners
        cv.drawChessboardCorners(image_cam3, chessboard_size, corners2_cam3, ret_cam3)
        cv.imshow('chessboard calibration', image_cam3)
        cv.waitKey(1000)

# Calibrate camera
ret_cam3, K_cam3, dist_cam3, rvecs_cam3, tvecs_cam3 = cv.calibrateCamera(obj_points_cam3, img_points_cam3 ,gray_image_cam3.shape[::-1], None, None)

# Save parameters into numpy file
np.save("./camera_calib_params_cam3/ret", ret_cam3)
np.save("./camera_calib_params_cam3/K", K_cam3)
np.save("./camera_calib_params_cam3/dist", dist_cam3)
np.save("./camera_calib_params_cam3/rvecs", rvecs_cam3)
np.save("./camera_calib_params_cam3/tvecs", tvecs_cam3)


# Outputs Camera Matrix and Distortion Coefficients
print('=Camera Matrix (K)=\n', K_cam3, '\n')
print('=Distortion Coefficients=\n', dist_cam3, '\n')
print('=Rotational Vectors=\n', rvecs_cam3, '\n')
print('=Translational Vectors=\n', tvecs_cam3, '\n')
print('=Ret Value=\n', ret_cam3, '\n')




img_cam3 = cv.imread('/Users/yasmeen/Desktop/side_project_cabin/ContinuumRoboticsLab/calibimgs_cam3_NEW2/cam3_1.png')
cv.imshow('cam3_1.png', img_cam3)

h_cam3,  w_cam3 = img_cam3.shape[:2]

# Returns the new camera intrinsic matrix based on the free scaling parameter.
newCameraMatrix_cam3, roi_cam3 = cv.getOptimalNewCameraMatrix(K_cam3, dist_cam3, (w_cam3 ,h_cam3), 1, (w_cam3 ,h_cam3))



# UNDISTORTION
# undistort function and use ROI obtained to crop the result with OpenCV.
dst = cv.undistort(img_cam3, K_cam3, dist_cam3, None, newCameraMatrix_cam3)

# cropping the image with the ROI
x_cam3, y_cam3, w_cam3, h_cam3 = roi_cam3
dst_cam3 = dst[y_cam3: y_cam3 +h_cam3, x_cam3: x_cam3 +w_cam3]
cv.imwrite('calibratedImg1.png', dst_cam3)

# Calculate projection error.
mean_error_cam3 = 0
for i in range(len(obj_points_cam3)):
    img_points2, _ = cv.projectPoints(obj_points_cam3[i] ,rvecs_cam3[i] ,tvecs_cam3[i], K_cam3, dist_cam3)
    error_cam3 = cv.norm(img_points_cam3[i], img_points2, cv.NORM_L2 ) /len(img_points2)
    mean_error_cam3 += error_cam3

# output final re-projection error
print( "total error: {}".format(mean_error_cam3 /len(obj_points_cam3)))



# ------------------------------------------------------------
# PREPROCESSING

print("======== PHASE 2: PREPROCESSING (FEATURE MATCHING & KEYPOINT DETECTION) ========")


img1 = cv.imread('aruco1_1_1.png', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('aruco3_3_3.png', cv.IMREAD_GRAYSCALE)


# cv.waitKey(0) # waits until a key is pressed
# cv.destroyAllWindows() # destroys the window showing image

# Compare unprocessed images (a visual)
fig, axes = plt.subplots(1, 2, figsize=(15, 10))
axes[0].imshow(img1, cmap="gray")
axes[1].imshow(img2, cmap="gray")

# img 1 lines (top & bottom bound respectively)
axes[0].axhline(855)
axes[0].axhline(1770)

# img 2 lines (top & bottom bound respectively)
axes[1].axhline(700)
axes[1].axhline(1650)
plt.suptitle("Original Images Comparison")
plt.show()


# 1. Detect keypoints and their descriptors
# Based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html

# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Visualize keypoints
imgSift = cv.drawKeypoints(
    img1, kp1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow("SIFT Keypoints", imgSift)
cv.imwrite("sift_keypoints.png", imgSift)


# Match keypoints in both images
# Based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)


# Keep good matches: calculate distinctive image features
# Lowe, D.G. Distinctive Image Features from Scale-Invariant Keypoints.
# https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
matchesMask = [[0, 0] for i in range(len(matches))]
good = []
pts1 = []
pts2 = []

for i, (m, n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        # Keep this keypoint pair
        matchesMask[i] = [1, 0]
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

# Draw the keypoint matches between both pictures
# Still based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask,
                   flags=cv.DrawMatchesFlags_DEFAULT)

keypoint_matches = cv.drawMatchesKnn(
    img1, kp1, img2, kp2, matches, None, **draw_params)
cv.imshow("Keypoint matches", keypoint_matches)
cv.imwrite("keypoint_matches.png", keypoint_matches)



# ------------------------------------------------------------

print("======== PHASE 3: EPIPOLAR GEOMETRY ========")

# Calculate the fundamental matrix for the cameras
# https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
fundamental_matrix, inliers = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)

# We select only inlier points
pts1 = pts1[inliers.ravel() == 1]
pts2 = pts2[inliers.ravel() == 1]


# Visualize epilines
# Adapted from: https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html


def drawlines(img1src, img2src, lines, pts1src, pts2src):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1src.shape
    img1color = cv.cvtColor(img1src, cv.COLOR_GRAY2BGR)
    img2color = cv.cvtColor(img2src, cv.COLOR_GRAY2BGR)
    # Edit: use the same random seed so that two images are comparable!
    np.random.seed(0)
    for r, pt1, pt2 in zip(lines, pts1src, pts2src):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1color = cv.line(img1color, (x0, y0), (x1, y1), color, 1)
        img1color = cv.circle(img1color, tuple(pt1), 5, color, -1)
        img2color = cv.circle(img2color, tuple(pt2), 5, color, -1)
    return img1color, img2color


# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv.computeCorrespondEpilines(
    pts2.reshape(-1, 1, 2), 2, fundamental_matrix)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(
    pts1.reshape(-1, 1, 2), 1, fundamental_matrix)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

plt.subplot(121), plt.imshow(img5)
plt.subplot(122), plt.imshow(img3)
plt.suptitle("Epilines in both images")
plt.savefig("epilines.png")
plt.show()



#---------------------------------------------------------- Stereo Rectification ----------------------------------------------------------#

print("======== PHASE 4: STEREO RECTIFICATION ========")

# Stereo rectification (uncalibrated variant)
# Adapted from: https://stackoverflow.com/a/62607343

# NOTE: using the uncalibrated method to get the H1, H2 parameters has no influence on the overall calibration status, just used to get those parameters

h1, w1 = img1.shape
h2, w2 = img2.shape
_, H1, H2 = cv.stereoRectifyUncalibrated(
    np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1)
)


# Rectify (undistort) the images and save them
# Adapted from: https://stackoverflow.com/a/62607343
img1_rectified = cv.warpPerspective(img1, H1, (w1, h1))
img2_rectified = cv.warpPerspective(img2, H2, (w2, h2))
cv.imwrite("rectified_1.png", img1_rectified)                       # FOR LOOP IT AND LOAD THE RECTIFIED IMAGES TO THE RECTFIFIED_IMGS_CAM# FOLDERS
cv.imwrite("rectified_2.png", img2_rectified)

# Draw the rectified images
fig, axes = plt.subplots(1, 2, figsize=(15, 10))
axes[0].imshow(img1_rectified, cmap="gray")
axes[1].imshow(img2_rectified, cmap="gray")
axes[0].axhline(670)
axes[1].axhline(670)
axes[0].axhline(1650)
axes[1].axhline(1650)
plt.suptitle("Rectified images")
plt.savefig("rectified_images.png")
plt.show()


# ---------------------------------------------------------- Detect ArUco (detected array of Aruco marker corners in each image) ----------------------------------------------------------#

print("======== PHASE 5: ARUCO MARKER DETECTION ========")


path_1 = r'/Users/yasmeen/Desktop/side_project_cabin/ContinuumRoboticsLab/rectified_1.png'
path_3 = r'/Users/yasmeen/Desktop/side_project_cabin/ContinuumRoboticsLab/rectified_2.png'

img_test_1 = cv.imread(path_1)
img_test_3 = cv.imread(path_3)


projPoints1 = np.array(detectAruco(img_test_1, markerSize=6, totalMarkers=250, draw=True))
projPoints3 = np.array(detectAruco(img_test_3, markerSize=6, totalMarkers=250, draw=True))


# transposing to get the desired 2xN projection matrix size (for the triangulatePoints method)
projPoints1 = np.transpose(projPoints1)
projPoints3 = np.transpose(projPoints3)


print(projPoints1)
print(projPoints1.shape)

print(projPoints3)
print(projPoints3.shape)



#---------------------------------------------------------- Projection Matrices ----------------------------------------------------------#

print("======== PHASE 6: PROJECTION MATRICES ========")



# projMatr1 & 2 are the calculated matrices separately for cams 1 & 3

# PROJECTION MATRICES (CAMERA 1)
rotation_mat_cam1 = np.zeros(shape=(3, 3))
R_cam1 = cv.Rodrigues(rvecs_cam1[0], rotation_mat_cam1)[0]
projMatr1 = np.column_stack((np.matmul(K_cam1,R_cam1), tvecs_cam1[0]))

# PROJECTION MATRICES (CAMERA 3)
rotation_mat_cam3 = np.zeros(shape=(3, 3))
R_cam3 = cv.Rodrigues(rvecs_cam3[0], rotation_mat_cam3)[0]
projMatr3 = np.column_stack((np.matmul(K_cam3,R_cam3), tvecs_cam3[0]))

print("Project Matrix 1")
print (projMatr1)


print("Project Matrix 3")
print (projMatr3)

# #---------------------------------------------------------- Triangulation ----------------------------------------------------------#
#

print("======== PHASE 7: TRIANGULATION & 3D COORDINATES ========")

# The function reconstructs 3-dimensional points (in homogeneous coordinates) by using their observations with a stereo camera.
final_coordinates = cv.triangulatePoints(projMatr1, projMatr3, projPoints1, projPoints3)

# 'points' is converted to un-homogeneous ( local ) coordinates by dividing 'x, y, z' with 'w', the 4th row
one_aruco_marker_corner_coordinate = final_coordinates/final_coordinates[3]

print("Final Triangulated Points Matrix")
print(final_coordinates)

print("One AruUco Marker Coordinate")
print(one_aruco_marker_corner_coordinate)


cv.waitKey()
cv.destroyAllWindows()
# ---------------------------------------------------------------
