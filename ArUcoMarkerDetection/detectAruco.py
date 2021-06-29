import cv2
import cv2.aruco as aruco
import numpy as np
import os


# from pathlib import Path
# # import glob
# # import os

def findArucoMarkers(img, markerSize=6, totalMarkers=250, draw=True):
    """
    Returns the found ArUco markers in a frame with details.
    :param img: the image where the aruco markers exist
    :param markerSize: the size of the markers
    :param totalMarkers: the total number of markers (in chosen ArUco DICT version)
    :param draw: the bboxes drawn around the detected markers
    :return: the detected bboxes and aruco id of the detected markers
    """
    # converting image to gray
    imgGray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # customized ArUco Dictionary key
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')

    # defining the ArUco Dictionary
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(imgGray,
                                               arucoDict,
                                               parameters=arucoParam)

    # prints ids of arUco markers detected
    # print(ids)

    # draws boundary boxes and ids around detected arUco markers
    if draw:
        aruco.drawDetectedMarkers(img, bboxs, ids)

    return [bboxs, ids]


def main():
    # opens live video camera
    # NOTE: built-in camera is 0, multiple cameras are scaled to 1, 2, 3, etc.
    cap = cv2.VideoCapture(0)

    while True:
        # loop continuously reading frame-by-frame
        success, frame = cap.read()
        if success:
            # find the aruco markers in the image frame (img/frame)
            findArucoMarkers(frame)
            # a frame was successfully read
            # show camera feed in a window
            cv2.imshow("Live Video", frame)

            # check for a key pressed event and break the camera loop
            k = cv2.waitKey(5) & 0xFF

            # click the escape button on keyboard to exit camera view
            if k == 27:
                # closes the webcam window
                cv2.destroyAllWindows()
                cap.release()
                break

        # frame wasn't read, handle that problem:
        else:
            # closes the webcam window
            cv2.destroyAllWindows()
            cap.release()
            break


# main program executor
if __name__ == "__main__":
    main()