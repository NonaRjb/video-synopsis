import argparse
import imutils
import cv2
import numpy as np
from tracker import Tracker
import background
import time


# add input arguments
ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', help='path to the video file')
ap.add_argument('-o', '--output', help='path to the output video file')
args = vars(ap.parse_args())

if args['video'] is None:
    vs = cv2.VideoCapture('Video1.avi')
else:
    vs = cv2.VideoCapture(args["video"])

start_time = time.time()
gray_medianFrame, bg = background.bg_estimate(vs, 50, (5, 5))

if args['video'] is None:
    vs = cv2.VideoCapture('Video1.avi')
else:
    vs = cv2.VideoCapture(args["video"])

# get frame per second (fps) of input video
fps = vs.get(cv2.CAP_PROP_FPS)
# initialize an object from Tracker class
ct = Tracker(fps, bg)   # max_disappeared = 20 for Video1 & 25 for Video2
frame_id = 0

while True:
    frame = vs.read()
    frame = frame[1]

    if frame is None:
        break

    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # subtract estimated background from current frame
    frameDelta = cv2.absdiff(gray_medianFrame, gray)
    # create binary mask from foreground objects
    thresh = cv2.threshold(frameDelta, 20, 255, cv2.THRESH_BINARY)[1]
    # apply median filter to remove salt and pepper noise
    thresh = cv2.medianBlur(thresh, 5)

    # apply some morphological operations to make connected objects
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    thresh = cv2.erode(thresh, None, iterations=1)   # 1 for Video1
    thresh = cv2.dilate(thresh, None, iterations=5)  # 5 for Video1

    # specify moving objects
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    rects = []
    masks = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        rects.append(np.array([x, y, x+w, y+h]).astype('int'))
        # generate binary masks for each moving object
        object_mask = np.zeros_like(frame)
        cv2.drawContours(object_mask, [c], 0, (255, 255, 255), cv2.FILLED)
        object_mask[object_mask > 0] = 255
        masks.append(object_mask)
    # update tracker
    objects = ct.update(rects, masks, frame_id, frame)
    frame_id += 1

vs.release()

ct.complete_last_frame(frame_id)
ct.set_object_times()

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

print("total time to execute the program --- %s seconds ---" % (time.time() - start_time))
