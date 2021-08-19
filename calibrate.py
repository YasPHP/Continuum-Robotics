
import cv2 as cv
import numpy as np
import glob
from tqdm import tqdm
from typing import List
import PIL.ExifTags
import PIL.Image

# ========================= CAMERA CALIBRATION ========================= #


class Camera:
    """A Camera.

    Create a camera object and run various chessboard calibrations on initialized, n cameras.

    === Attributes ===
    chessboard_size: the (row, col) dimensions
    obj_points: 3D points in real world space
    img_points: 3D points in image plane
    objp: points to display on chessboard grid
    calibration_img_paths: path of the camera image dataset
    """

    # Attribute types
    chessboard_size: tuple
    obj_points: List[int]
    img_points: List[int]
    objp: List[int]
    calibration_img_paths: List[str]


    def __init__(self, name: str, chessboard_size: tuple, calibration_img_paths: glob) -> None:
        """
        Initialize a new Camera object.
        :param name: name/type/number of camera
        :param chessboard_size: the (row, col) dimensions
        :param calibration_img_paths: path of the camera image dataset

        >>> cam1 = Camera("camera 1", (9 ,6), [], [], [], glob.glob('./calibimgs_cam3/*'))
        >>> cam1.name
        "camera 1"
        >>> cam1.chessboard_size()
        (9 ,6)
        >>> cam1.obj_points()
        []
        >>> cam1.img_points()
        []
        >>> cam1.objp()
        np.zeros((np.prod(chessboard_size) ,3) ,dtype=np.float32)
        >>> cam1.calibration_img_paths()
        glob.glob('./calibimgs_cam3/*')
        """

        self.name = name

        self.chessboard_size = chessboard_size # Define size of chessboard target

        # Define arrays to save detected points
        self.obj_points = [] # 3D points in real world space
        self.img_points = [] # 3D points in image plane

        # Prepare grid and points to display
        self.objp = np.zeros((np.prod(chessboard_size) ,3) ,dtype=np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

        # the folder path of the calibration images
        self.calibration_img_paths = calibration_img_paths



    def chessboardDetection(self, calibration_img_paths: glob):
        """
        Detects the chessboard in an image frame.

        :param calibration_paths: the explicit path for the chessboard camera calibration dataset
        """

        # Iterate over images to find intrinsic matrix
        for image_path in tqdm(calibration_img_paths):

            # Load image
            image = cv.imread(image_path)
            gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            print("Image loaded, Analyzing...")
            # find chessboard corners
            ret ,corners = cv.findChessboardCorners(gray_image, self.chessboard_size, None)

            if ret == True:
                print("Chessboard detected!")
                print(image_path)
                # define criteria for subpixel accuracy
                criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                # refine corner location (to subpixel accuracy) based on criteria.
                corners2 = cv.cornerSubPix(gray_image, corners, (5 ,5), (-1 ,-1), criteria)
                self.obj_points.append(self.objp)
                self.img_points.append(corners)

                # Draw and display the corners
                cv.drawChessboardCorners(image, self.chessboard_size, corners2, ret)
                cv.imshow('chessboard calibration', image)
                cv.waitKey(1000)


    def calibrateCamera(self, gray_image):
        """
        Calibrates the camera and saves the extrinsic and intrinsic calibration parameters to external numpy files to be later retrieved.

        :param gray_image: the image to be calibrated
        """

        # Calibrate camera
        ret, K, dist, rvecs, tvecs = cv.calibrateCamera(self.obj_points, self.img_points ,gray_image.shape[::-1], None, None)

        # Save parameters into external numpy files in the camera_calib_params folder
        np.save("./camera_calib_params/ret", ret)
        np.save("./camera_calib_params/K", K)
        np.save("./camera_calib_params/dist", dist)
        np.save("./camera_calib_params/rvecs", rvecs)
        np.save("./camera_calib_params/tvecs", tvecs)


        # Outputs Camera Matrix and Distortion Coefficients
        print('=Camera Matrix (K)=\n', K, '\n')
        print('=Distortion Coefficients=\n', dist, '\n')
        print('=Rotational Vectors=\n', rvecs, '\n')
        print('=Translational Vectors=\n', tvecs, '\n')
        print('=Ret Value=\n', ret, '\n')



def calculateOptimalNewCameraMatrix(self, img, K: List, dist: List):
    """
    Returns the new camera intrinsic matrix based on the free scaling parameter.

    :param img: the camera image to be considered
    :param K: the camera matrix
    :param dist: the distance
    """

    # Load one of the test images (picking the first image in each camera calibration dataset)
    img = cv.imread('/Users/yasmeen/Desktop/side_project_cabin/ContinuumRoboticsLab/calibimgs_cam3/pic1_cam3.png')
    cv.imshow('pic1_cam3.png', img)

    h,  w = img.shape[:2]

    # Returns the new camera intrinsic matrix based on the free scaling parameter.
    # If the scaling parameter alpha=0, it returns undistorted image with minimum unwanted pixels.
    # So it may even remove some pixels at image corners.
    # If alpha=1, all pixels are retained with some extra black images.
    # It also returns an image ROI which can be used to crop the result.
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(K, dist, (w ,h), 1, (w ,h))



    def undistortCamera(self, img, K: List, dist: List, rvecs: List, tvecs: List):
        """
        Returns undistorted camera images with a built-in OpenCV method

        :param img: the image to be undisorted
        :param K: the inputted camera matrix
        :param dist: the distance
        :param rvecs: rotation vector matrix
        :param tvecs: translational vector matrix
        """

        # call undistort function and use ROI obtained to crop the result with simple OpenCV method
        dst = cv.undistort(img, K, dist, None, newCameraMatrix)

        # cropping the image with the ROI
        x, y, w, h = roi
        dst = dst[y: y +h, x: x +w]
        cv.imwrite('calibratedImg1.png', dst)

        # Calculate projection error.
        mean_error = 0
        for i in range(len(self.obj_points)):
            img_points2, _ = cv.projectPoints(self.obj_points[i] ,rvecs[i] ,tvecs[i], K, dist)
            error = cv.norm(self.img_points[i], img_points2, cv.NORM_L2 ) /len(img_points2)
            mean_error += error

        # outputs final re-projection error
        print( "total error: {}".format(mean_error /len(self.obj_points)))



"""
Calibration Output Results:

=Camera Matrix (K)=
 [[532.82710711   0.         342.48678547]
 [  0.         532.94588592 233.85594816]
 [  0.           0.           1.        ]] 

=Distortion Coefficients=
 [[-2.80880910e-01  2.51716562e-02  1.21657346e-03 -1.35549404e-04
   1.63448912e-01]] 

total error: 0.02641766256162398
"""

