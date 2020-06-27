import argparse
import imutils
import cv2
import numpy as np
from centroidtracker import CentroidTracker

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', help='path to the video file')
args = vars(ap.parse_args())

vs = cv2.VideoCapture(args["video"])
first_frame = None

# Randomly select 25 frames
frameIds = vs.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=50)

frames = []
for fid in frameIds:
    vs.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = vs.read()
    frame = imutils.resize(frame, width=500)
    frames.append(frame)

vs.release()
# Calculate the median along the time axis
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
gray_medianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
gray_medianFrame = cv2.GaussianBlur(gray_medianFrame, (7, 7), 0)

cv2.imshow('background', gray_medianFrame)

vs = cv2.VideoCapture(args["video"])
ct = CentroidTracker()
prev_objects = None
color = np.random.randint(0, 255, (100, 3))
mask = np.zeros_like(frames[0])
prev_frame = cv2.absdiff(gray_medianFrame, cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY))

while True:
    frame = vs.read()
    frame = frame[1]

    if frame is None:
        break

    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    '''if first_frame is None:
        first_frame = gray
        cv2.imshow('First Frame', first_frame)
        continue'''

    frameDelta = cv2.absdiff(gray_medianFrame, gray)
    thresh = cv2.threshold(frameDelta, 30, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.dilate(thresh, None, iterations=5)
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
        rects.append(np.array([x, y, x+w, y+h]).astype('int'))
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    objects = ct.update(rects)

    '''for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)'''

    i = 0
    if prev_objects is not None:
        for (objectID, centroid) in objects.items():
            for (prev_objectID, prev_centroid) in prev_objects.items():
                if prev_objectID == objectID:
                    a, b = centroid.ravel()
                    c, d = prev_centroid.ravel()
                    mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                    frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
                    text = "ID {}".format(objectID)
                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    i += 1

    prev_objects = objects
    prev_frame = frameDelta

    frame = cv2.add(frame, mask)
    cv2.imshow("Security Feed", frame)
    cv2.imshow("Thresh", thresh)
    # cv2.imshow("Frame Delta", frameDelta)
    key = cv2.waitKey(30) & 0xFF

    if key == ord("q"):
        break


vs.release()
cv2.destroyAllWindows()