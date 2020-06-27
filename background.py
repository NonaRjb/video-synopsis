import cv2
import imutils
import numpy as np


def bg_estimate(cap, sample_num, kernel_size):

    frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=sample_num)

    frames = []
    for fid in frameIds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=500)
        frames.append(frame)

    cap.release()
    medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
    gray_medianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
    gray_medianFrame = cv2.GaussianBlur(gray_medianFrame, kernel_size, 0)

    return gray_medianFrame, medianFrame
