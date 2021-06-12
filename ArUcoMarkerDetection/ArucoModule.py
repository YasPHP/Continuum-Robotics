import cv2
import cv2.aruco as aruco
import numpy as np
import os


def main():
    # opens live video camera
    # NOTE: built-in camera is 0, multiple cameras are scaled to 1, 2, 3, etc.
    cap = cv2.VideoCapture(0)

    while True:
        # loop continuously reading frame-by-frame
        success, frame = cap.read()
        if success:
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