import argparse
import imutils
import cv2
import numpy as np
from tracker import Tracker
import background


ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', help='path to the video file')
ap.add_argument('-o', '--output', help='path to the output video file')
args = vars(ap.parse_args())

if args['video'] is None:
    vs = cv2.VideoCapture('Video1.avi')
else:
    vs = cv2.VideoCapture(args["video"])

gray_medianFrame, bg = background.bg_estimate(vs, 50, (5, 5))

if args['video'] is None:
    vs = cv2.VideoCapture('Video1.avi')
else:
    vs = cv2.VideoCapture(args["video"])

fps = vs.get(cv2.CAP_PROP_FPS)
ct = Tracker(fps, bg)
prev_objects = None
color = np.random.randint(0, 255, (100, 3))
mask = None
first_frame = None
frame_id = 0

while True:
    frame = vs.read()
    frame = frame[1]

    if frame is None:
        break

    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if first_frame is None:
        mask = np.zeros_like(frame)
        first_frame = frame

    frameDelta = cv2.absdiff(gray_medianFrame, gray)
    thresh = cv2.threshold(frameDelta, 20, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.medianBlur(thresh, 5)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    thresh = cv2.erode(thresh, None, iterations=1)   # 1 for Video1
    thresh = cv2.dilate(thresh, None, iterations=5)  # 5 for Video1

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    rects = []
    masks = []

    # cv2.drawContours(frame, cnts, -1, (0, 0, 255), 1)
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        rects.append(np.array([x, y, x+w, y+h]).astype('int'))

        object_mask = np.zeros_like(frame)
        # object_mask[y:y+h, x:x+w] = 255
        cv2.drawContours(object_mask, [c], 0, (255, 255, 255), cv2.FILLED)
        object_mask[object_mask > 0] = 255
        masks.append(object_mask)
        # out = np.zeros_like(frame)  # Extract out the object and place into output image
        # out[object_mask == 255] = frame[object_mask == 255]
        # cv2.imshow('out', out)
        # cv2.waitKey(20) & 0xFF
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    objects = ct.update(rects, masks, frame_id, frame)

    '''for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)'''

    '''i = 0
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

    frame = cv2.add(frame, mask)'''
    # cv2.imshow("Security Feed", frame)
    # cv2.imshow("Thresh", thresh)
    # cv2.imshow("Frame Delta", frameDelta)
    # key = cv2.waitKey(30) & 0xFF

    # if key == ord("q") or key == 27:
    #     break

    frame_id += 1

vs.release()
cv2.destroyAllWindows()

ct.complete_last_frame(frame_id)
ct.set_object_times()
# ct.get_time()

if args['output'] is None:
    out_file = 'project.avi'
else:
    out_file = args['output']

summarized_video = ct.get_video()
height, width, _ = summarized_video[0].shape
size = (width, height)
out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

for i in range(len(summarized_video)):
    out.write(summarized_video[i])
out.release()