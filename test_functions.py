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
bboxs = []

# ======= IMAGE PREPROCESSING PARAMETERS ======= #
# chose camera 1 & 3 image dataset from lab b/c 46.08% overlap similarity
img_path_cam1 = glob.glob('./calibimgs_cam1_NEW/*')
img_path_cam3 = glob.glob('./calibimgs_cam3_NEW/*')

img1 = cv.imread(img_path_cam1[-5], cv.IMREAD_GRAYSCALE)    # 7th pic
img2 = cv.imread(img_path_cam3[-6], cv.IMREAD_GRAYSCALE)    # 7th pic


# ======= STEREO RECTIFICATION PARAMETERS ======= #
good = []
pts1 = []
pts2 = []


# ======== PHASE 2: PREPROCESSING (FEATURE MATCHING & KEYPOINT DETECTION) ======== #

def compareOrigImgs(img_path_cam1: str, img_path_cam3:str):
    """
    Returns a matplot comparing the position alignment of the ArUco boards
    in the unprocessed images of two different cameras.

    :param img_path_cam1: the folder path of the camera 1 images
    :param img_path_cam3: the folder path of the camera 3 images

    # BASE CASE TESTING DOC TESTS (run these two)
    compareOrigImgs(img_path_cam1, img_path_cam3)
    """

    # iterates through both camera version's pics simultaneously
    for i in range(1, 15):
        img1 = cv.imread(f'calibimgs_cam1_NEW/{i}.png', cv.IMREAD_GRAYSCALE)
        img2 = cv.imread(f'calibimgs_cam3_NEW/{i}.png', cv.IMREAD_GRAYSCALE)

        # cv.waitKey(0) # waits until a key is pressed
        # cv.destroyAllWindows() # destroys the window showing image

        # unprocessed images are compared (a visual)
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
    cv.imshow("SIFT Keypoints Camera 1", img_sift_1)
    cv.imwrite("sift_keypoints_1.png", img_sift_1)

    cv.imshow("SIFT Keypoints Camera 2", img_sift_2)
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
    # https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

    # additional mask to filter through matches
    matchesMask = [[0, 0] for i in range(len(matches))]

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            # choose this keypoint pair out of others
            matchesMask[i] = [1, 0]
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    # sraw the keypoint matches between both pictures
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv.DrawMatchesFlags_DEFAULT)

    keypoint_matches = cv.drawMatchesKnn(
        img1, kp1, img2, kp2, matches, None, **draw_params)

    cv.imshow("Keypoint Matches between Cameras", keypoint_matches)
    cv.imwrite("keypoint_matches.png", keypoint_matches)



# ======== PHASE 3: EPIPOLAR GEOMETRY ======== #

def calculateFundamentalMatrix(pts1: List, pts2: List):
    """
    Calculates the fundamental matrix for the cameras based off the keypoint pairs between the two cameras.

    :param pts1: keypoint pair from camera 1.
    :param pts2: keypoint pair from camera 2.

    # BASE CASE TESTING DOC TESTS (run these two)
    matchDetectedKeypoints(img1, img2)
    calculateFundamentalMatrix(pts1, pts2)
    """

    # Calculate the fundamental matrix for the cameras
    # https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    fundamental_matrix, inliers = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)

    # Only inlier points are selected
    pts1 = pts1[inliers.ravel() == 1]
    pts2 = pts2[inliers.ravel() == 1]

    print(fundamental_matrix)
    return fundamental_matrix




def drawlines(img1src, img2src, lines, pts1src, pts2src):
    """
    Returns and visualizes the epilines between the two camera photos.
    Adapted from: https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html

    :param img1src: image on which we draw the epilines for the points in img2
    :param img2src: image on which we draw the epilines for the points in img1
    :param lines: corresponding epilines
    :param pts1src: source of points 1 in camera 1
    :param pts2src: source of points 2 in camera 2
    :return: img1color, img2color
    """
    # deduce camera image shape
    r, c = img1src.shape

    # images converted colour
    img1color = cv.cvtColor(img1src, cv.COLOR_GRAY2BGR)
    img2color = cv.cvtColor(img2src, cv.COLOR_GRAY2BGR)

    # note to use the same random seed so that the two camera images can be compared
    np.random.seed(0)

    #Epiline generating algorithm adopted from OpenCV
    for r, pt1, pt2 in zip(lines, pts1src, pts2src):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1color = cv.line(img1color, (x0, y0), (x1, y1), color, 1)
        img1color = cv.circle(img1color, tuple(pt1), 5, color, -1)
        img2color = cv.circle(img2color, tuple(pt2), 5, color, -1)
    return img1color, img2color


def computeEpilines(fundamental_matrix: List):
    """
    Computes and visualizes the epilines corresponding to points in both images.

    :param fundamental_matrix: the matrix to base the 3D epipolar geometry on
    """

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

    # matplot graph outlining matched epilines!
    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)
    plt.suptitle("Epilines Detected in Both Camera Images")
    plt.savefig("epilines.png")
    plt.show()




# ======== PHASE 4: STEREO RECTIFICATION ======== #

def rectifyImgs(img1: str, img2: str, fundamental_matrix: List):
    """
    Returns stereo rectified image pairs to a common plane and visualizes them.

    :param img1: image from camera 1
    :param img2: image from camera 2
    :param fundamental_matrix: the matrix to base the 3D epipolar geometry on
    """

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
    cv.imwrite("rectified_1.png", img1_rectified)
    cv.imwrite("rectified_2.png", img2_rectified)

    # Draws the rectified images in a matplot graph
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    axes[0].imshow(img1_rectified, cmap="gray")
    axes[1].imshow(img2_rectified, cmap="gray")
    axes[0].axhline(1190)
    axes[1].axhline(1190)
    axes[0].axhline(2950)
    axes[1].axhline(2950)
    plt.suptitle("Rectified images")
    plt.savefig("rectified_images.png")
    plt.show()



# ======== PHASE 5: ARUCO MARKER DETECTION ======== #

def iterateImg():   # (img_folder_path: str): the folder path of the images where the aruco markers exist
    """
    Returns the found ArUco markers in multiple frames through iteration.


    # BASE CASE TESTING DOC TESTS (run these two)
    img_folder_path = glob.glob('./aruco_imgs/*')
    iterateImg(img_folder_path)
    """

    # iterates through both camera version's pics simultaneously (+1 to end of range b/c non-inclusive)
    for i in range(1, 5):

        # image paths from both cameras, where the arUco markers are to be detected
        img1 = cv.imread(f'rectified_imgs_cam1/{i}.png')
        img2 = cv.imread(f'rectified_imgs_cam3/{i}.png')

        # # TEMPORARY PLACEHOLDER (W/ NON-RECTIFIED IMAGES) FOR ABOVE
        # img1 = cv.imread(f'arucoimgs_cam1/{i}.png')
        # img2 = cv.imread(f'arucoimgs_cam3/{i}.png')

        # projPoints1 & 3 are the detected array of Aruco marker corners in each image
        projPoints1 = detectAruco(img1, markerSize=6, totalMarkers=250, draw=True)
        projPoints3 = detectAruco(img2, markerSize=6, totalMarkers=250, draw=True)

        print(projPoints1)
        print(projPoints3)


def detectAruco(img: str, markerSize: int, totalMarkers: int, draw: bool):
    """
    Returns the four corners of each detected ArUco marker in a frame and draws boundary boxes with an arUco ID.

    :param img: the image where the aruco markers exist
    :param markerSize: the size of the markers
    :param totalMarkers: the total number of markers (in chosen ArUco DICT version)
    :param draw: the bboxes drawn around the detected markers
    :return: the four detected corners of each detected marker and draws bboxes with an aruco id

    # BASE CASE TESTING DOC TESTS (run these two)
    img = cv.imread('aruco markers plate.png')
    detectAruco(img, markerSize=6, totalMarkers=250, draw=True)
    """

    # key to exit the window view
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


    # prints the arUco marker's detected bounding box and id on the img window directly
    print([bboxs, ids])

    # outputs a window with the detected aruco markers
    cv.imshow("ArUco Marker Detection", detected_markers)
    cv.waitKey(0)


    # gets the four corners of the aruco marker
    top_left = bboxs[0][0][0] # bboxs[0][0][1]
    top_right = bboxs[0][0][1] # bboxs[0][1][1]
    bottom_right = bboxs[0][0][2] # bboxs[0][2][1]
    bottom_left = bboxs[0][0][3] # bboxs[0][3][1]



    # ouputs the coordinates of each detected arUco marker's four corners
    return [top_left, top_right, bottom_right, bottom_left]





# ======== PHASE 6: PROJECTION MATRICES ======== #


def calculateProjectionMatrix(K_cam1: List, rvecs_cam1: List, tvecs_cam1: List, K_cam3: List, rvecs_cam3: List, tvecs_cam3: List):
    """
    Calculates the projection matrix for each camera via their intrinsic & extrinsic calibration parameters.

    :param K_cam1: camera 1 matrix
    :param rvecs_cam1: camera 1 rotational vectors
    :param tvecs_cam1: camera 1 translational vectors
    :param K_cam3: camera 3 matrix
    :param rvecs_cam3: camera 3 rotational vectors
    :param tvecs_cam3: camera 3 translational vectors
    """


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




# ======== PHASE 7: TRIANGULATION & 3D COORDINATES ======== #



def triangulate3DPoints(projMatr1: List, projMatr3: List, projPoints1: List, projPoints3: List):
    """
    Reconstructs 3-dimensional points (in homogeneous coordinates) by using their observations with a stereo camera.

    :param projMatr1:
    :param projMatr3:
    :param projPoints1:
    :param projPoints3:
    :return:
    """
    cv.triangulatePoints(projMatr1, projMatr3, projPoints1, projPoints3)




# main program executor
if __name__ == "__main__":
    # matchDetectedKeypoints(img1, img2)
    # compareOrigImgs(img_path_cam1, img_path_cam3)

    # img_folder_path = glob.glob('./TDCR_imgs/*')
    # iterateImg(img_folder_path)

    # img = cv.imread('sift_keypoints_1.png')
    # detectAruco(img, markerSize=6, totalMarkers=250, draw=True)

    # img = cv.imread('rectified_1.png')
    # detectAruco(img, markerSize=6, totalMarkers=250, draw=True)

    # img = cv.imread('rectified_1.png')
    # detectAruco(img, markerSize=6, totalMarkers=250, draw=True)


    path_1 = r'/Users/yasmeen/Desktop/side_project_cabin/ContinuumRoboticsLab/aruco_imgs_final/aruco2_2_2.png'
    path_2 = r'/Users/yasmeen/Desktop/side_project_cabin/ContinuumRoboticsLab/aruco_imgs_final/aruco3_3_3.png'

    img1 = cv.imread(path_1)
    img2 = cv.imread(path_2)
    matchDetectedKeypoints(img1, img2)


