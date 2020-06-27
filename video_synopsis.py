import numpy as np
import cv2

cap = cv2.VideoCapture('Video1.avi')

subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = subtractor.apply(gray)
    # mask = cv2.medianBlur(mask, 5)
    # mask = cv2.GaussianBlur(mask, (5, 5), -1)
    mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.erode(mask, None, iterations=3)

    mask = cv2.dilate(mask, None, iterations=5)
    # thresh = cv2.erode(thresh, None, iterations=2)

    cv2.imshow('frame', mask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
