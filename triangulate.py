import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import os
import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
from typing import List, Callable

# ======= ARUCO PARAMETERS =======#
img_folder_path = glob.glob('./aruco_imgs/*')
markerSize=6
totalMarkers=250
draw=True
bboxs = [] # DOUBLE CHECK DUPLICACY

# ======= IMAGE PREPROCESSING PARAMETERS ======= #
# chose camera 1 & 3 image dataset from lab b/c 46.08% overlap similarity
img_path_cam1 = glob.glob('./calibimgs_cam1/*')
img_path_cam3 = glob.glob('./calibimgs_cam3/*')

# GONNA HAVE TO FIX THIS PARAM AND SEE HOW IT PLAYS OUT AS A GLOBAL OR LOCAL VARIABLE IN SOME ALL-ENCOMPASSING FUNCTION B/C IT'S REQUIRED 4 SEVERAL FUNCTIONS
img1 = cv.imread(img_path_cam1[-5], cv.IMREAD_GRAYSCALE)    # 7th pic
img2 = cv.imread(img_path_cam3[-6], cv.IMREAD_GRAYSCALE)    # 7th pic


# ======= STEREORECTIFICATION PARAMETERS ======= #
good = []
pts1 = []
pts2 = []




# ========================= PREPROCESSING ========================= #

def compareOrigImgs(img_path_cam1: str, img_path_cam3:str):
    """
    Returns a matplot comparing the position alignment of the ArUco boards
    in the unprocessed images of two different cameras.

    :param img_path_cam1: the folder path of the camera 1 images
    :param img_path_cam3: the folder path of the camera 3 images

    # BASE CASE TESTING DOC TESTS (run these two)
    compareOrigImgs(img_path_cam1, img_path_cam3)
    """

    img1 = cv.imread(img_path_cam1[-5], cv.IMREAD_GRAYSCALE)    # 7th pic
    img2 = cv.imread(img_path_cam3[-6], cv.IMREAD_GRAYSCALE)    # 7th pic

    # cv.imshow("Camera 1 (Image 1)", img1)
    # cv.imshow("Camera 3 (Image 1)", img2)

    # cv.waitKey(0) # waits until a key is pressed
    # cv.destroyAllWindows() # destroys the window showing image

    # Compare unprocessed images (a visual)
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    axes[0].imshow(img1, cmap="gray")
    axes[1].imshow(img2, cmap="gray")

    # img 1 lines (top & bottom bound respectively)
    axes[0].axhline(400)
    axes[0].axhline(850)

    # img 2 lines (top & bottom bound respectively)
    axes[1].axhline(340)
    axes[1].axhline(900)
    plt.suptitle("Original Images Comparison")
    plt.show()


def matchDetectedKeypoints(img1: str, img2: str):
    """
    Returns matched detected keypoints and their descriptors using
    the SIFT (Scale-invariant Feature Transform) algorithm and
    FLANN (Fast Library for Approximate Nearest Neighbors) based matcher.
    Based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html.

    :param img1: the image from camera 1
    :param img2: the image from camera 2
    """

    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Visualize keypoints (for both camera's images)
    img_sift_1 = cv.drawKeypoints(
        img1, kp1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    img_sift_2 = cv.drawKeypoints(
        img2, kp2, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Outputs the SIFT detected keypoints (for both camera's images)
    cv.imshow("SIFT Keypoints 1", img_sift_1)
    cv.imwrite("sift_keypoints_1.png", img_sift_1)

    cv.imshow("SIFT Keypoints 2", img_sift_2)
    cv.imwrite("sift_keypoints_2.png", img_sift_2)


    # Match keypoints in both images
    # Based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Keep good matches: calculate distinctive image features
    # Lowe, D.G. Distinctive Image Features from Scale-Invariant Keypoints.
    # International Journal of Computer Vision 60, 91–110 (2004). https://doi.org/10.1023/B:VISI.0000029664.99615.94
    # https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

    matchesMask = [[0, 0] for i in range(len(matches))]

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            # Keep this keypoint pair
            matchesMask[i] = [1, 0]
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    # Draw the keypoint matches between both pictures
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv.DrawMatchesFlags_DEFAULT)

    keypoint_matches = cv.drawMatchesKnn(
        img1, kp1, img2, kp2, matches, None, **draw_params)
    cv.imshow("Keypoint Matches", keypoint_matches)
    cv.imwrite("keypoint_matches.png", keypoint_matches)



# ========================= STEREORECTIFICATION ========================= #


def calculateFundamentalMatrix(pts1: List, pts2: List):
    """
    Calculates the fundamental matrix for the cameras.

    :param pts1:
    :param pts2:
    """

    # Calculate the fundamental matrix for the cameras
    # https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    fundamental_matrix, inliers = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)

    # We select only inlier points
    pts1 = pts1[inliers.ravel() == 1]
    pts2 = pts2[inliers.ravel() == 1]

    return fundamental_matrix




def drawlines(img1src, img2src, lines, pts1src, pts2src):
    """
    Visualizes epilines in the two camera photos.
    Adapted from: https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html

    :param img1src: image on which we draw the epilines for the points in img2
    :param img2src:
    :param lines: corresponding epilines
    :param pts1src:
    :param pts2src:
    :return: img1color, img2color
    """
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


def computeEpilines(fundamental_matrix: Callable):
    """
    Compute and find the epilines corresponding to points in both images.
    """

    # might not need- based on typehint (DOUBLE CHECK!!!!)
    fundamental_matrix = calculateFundamentalMatrix(pts1, pts2)

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




# # CONTINUE!!!!!!!!!!!!!!!!!!!!
#
# # Stereo rectification (uncalibrated variant)
# # Adapted from: https://stackoverflow.com/a/62607343
# h1, w1 = img1.shape
# h2, w2 = img2.shape
# _, H1, H2 = cv.stereoRectifyUncalibrated(
#     np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1)
# )


# do camera projection matrix calibration calaculations here (instead of above) : get rvecs calcs from camera calibration (so do this first for
# each camera and then input to get these 2 params: https://stackoverflow.com/questions/16101747/how-can-i-get-the-camera-projection-matrix-out-of-calibratecamera-return-value )






# # Rectify (undistort) the images and save them
# # Adapted from: https://stackoverflow.com/a/62607343
# img1_rectified = cv.warpPerspective(img1, H1, (w1, h1))
# img2_rectified = cv.warpPerspective(img2, H2, (w2, h2))
# cv.imwrite("rectified_1.png", img1_rectified)
# cv.imwrite("rectified_2.png", img2_rectified)
#
# # Draw the rectified images
# fig, axes = plt.subplots(1, 2, figsize=(15, 10))
# axes[0].imshow(img1_rectified, cmap="gray")
# axes[1].imshow(img2_rectified, cmap="gray")
# axes[0].axhline(1190)
# axes[1].axhline(1190)
# axes[0].axhline(2950)
# axes[1].axhline(2950)
# plt.suptitle("Rectified images")
# plt.savefig("rectified_images.png")
# plt.show()






# ========================= ARUCO DETECTION ========================= #

def iterateImg(img_folder_path: str):
    """
    Returns the found ArUco markers in multiple frames through iteration

    :param img_folder_path: the folder path of the images where the aruco markers exist

    # BASE CASE TESTING DOC TESTS (run these two)
    img_folder_path = glob.glob('./aruco_imgs/*')
    """

    # Iterate over images to find intrinsic matrix
    for image_path in tqdm(img_folder_path):
        # Load image
        image = cv.imread(image_path)
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        detectAruco(image, markerSize, totalMarkers, draw)



def detectAruco(img: str, markerSize: int, totalMarkers: int, draw: bool):
    """
    Returns the found ArUco markers in a frame with details.

    :param img: the image where the aruco markers exist
    :param markerSize: the size of the markers
    :param totalMarkers: the total number of markers (in chosen ArUco DICT version)
    :param draw: the bboxes drawn around the detected markers
    :return: the detected bboxes and aruco id of the detected markers

    # BASE CASE TESTING DOC TESTS (run these two)
    img = cv.imread('aruco markers plate.png')
    detectAruco(img, markerSize=6, totalMarkers=250, draw=True)
    """

    # to exit the window view
    ESC_KEY = 27

    # converting image to gray
    gray_image= cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # customized ArUco Dictionary key
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')  # using DICT_6X6_250

    # defining the ArUco Dictionary
    aruco_dict = aruco.Dictionary_get(key)
    aruco_params = aruco.DetectorParameters_create()
    # detecting the markers
    bboxs, ids, rejected = aruco.detectMarkers(gray_image,
                                               aruco_dict,
                                               parameters=aruco_params)

    # draws boundary boxes around detected arUco markers
    if draw:
        detected_markers = aruco.drawDetectedMarkers(img, bboxs, ids)

    print([bboxs, ids])

    # outputs a window with the detected aruco markers
    cv.imshow("ArUco Marker Detection",detected_markers)
    cv.waitKey(0)


    # check for a key pressed event and break the camera loop
    k = cv.waitKey(5) & 0xFF

    # click the escape button on keyboard to exit camera view
    if k == ESC_KEY:
        # closes the webcam window
        cv.destroyAllWindows()

    # frame wasn't read, handle that problem:
    else:
        # closes the webcam window
        cv.destroyAllWindows()




def getArucoCoordinates(bboxs, id, img):
    """
    Gets the coordinates of each detected arUco marker in an img.

    :param bbox: the boundary box of the aruco marker (four corner points)
    :param id: the id of the overlaid image to be displayed
    :param img: the image that will be drawn on top of
    :param imgAug: the displayed image over top the aruco marker
    :param drawId: the id displayed over top the aruco marker
    :return: the image frame with the augmented image overlaid
    """

    # gets the four corners of the aruco marker
    top_left = bboxs[0][0][0], bboxs[0][0][1]
    top_right = bboxs[0][1][0], bboxs[0][1][1]
    bottom_right = bboxs[0][2][0], bboxs[0][2][1]
    bottom_left = bboxs[0][3][0], bboxs[0][3][1]





# main program executor
if __name__ == "__main__":
    # matchDetectedKeypoints(img1, img2)
    # compareOrigImgs(img_path_cam1, img_path_cam3)

    img_folder_path = glob.glob('./TDCR_imgs/*')
    iterateImg(img_folder_path)


