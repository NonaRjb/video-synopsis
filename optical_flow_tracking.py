import argparse
import cv2
import imutils
import numpy as np
from centroidtracker import CentroidTracker

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', help='path to the video file')
args = vars(ap.parse_args())

vs = cv2.VideoCapture(args["video"])

while True:
    frame = vs.read()
    frame = frame[1]

    if frame is None:
        break

    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    frameDelta = cv2.absdiff(gray_medianFrame, gray)
    thresh = cv2.threshold(frameDelta, 35, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    rects = []

    for c in cnts:
        # if the contour is too small, ignore it
        # if cv2.contourArea(c) < args["min_area"]:
        #     continue
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        rects.append(np.array([x, y, x + w, y + h]).astype('int'))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if prev_objects is None:
        objects = ct.update(rects)
    else:
        objects = ct.update(rects)

    cv2.imshow("Security Feed", frame)
    key = cv2.waitKey(100) & 0xFF

    if key == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()
