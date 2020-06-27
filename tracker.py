from scipy.spatial import distance as dist
from collections import OrderedDict
from moving_object import MovingObject
import numpy as np
import cv2
import imutils


class Tracker:
    def __init__(self, fps, bg, maxDisappeared=20):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.moving_objects = OrderedDict()
        self.objects_seq = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.bg = bg
        self.video = []
        self.fps = fps

    def register(self, centroid, frame_num, mask):
        self.moving_objects[self.nextObjectID] = MovingObject(frame_num, centroid)
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.objects_seq[self.nextObjectID] = 0
        thresh = cv2.threshold(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        if len(self.video) == 0:
            print('len = 0')
            copy_bg = self.bg.copy()
            cv2.fillConvexPoly(copy_bg, np.squeeze(cnts[0]).astype(int), 0)
            self.video.append(cv2.add(copy_bg, mask))
            # # cv2.imshow('v', cv2.add(copy_bg, mask))
            # # cv2.waitKey(0)
        else:
            cv2.fillConvexPoly(self.video[self.objects_seq[self.nextObjectID]], np.squeeze(cnts[0]).astype(int), 0, 16)
            self.video[self.objects_seq[self.nextObjectID]] = cv2.add(self.video[self.objects_seq[self.nextObjectID]], mask)
            # cv2.imshow('w', self.video[self.objects_seq[self.nextObjectID]])
            # cv2.waitKey(0)
        self.objects_seq[self.nextObjectID] += 1
        self.nextObjectID += 1

    def deregister(self, objectID, frame_num):
        self.moving_objects[objectID].set_last_frame(frame_num-self.disappeared[objectID])
        self.moving_objects[objectID].set_time(self.fps)
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects, masks, frame_num, frame):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                # self.moving_objects[objectID].set_mask(np.zeros_like(frame))
                self.objects_seq[objectID] += 1
                # self.video.append(self.bg.copy())
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID, frame_num)

            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                mask = np.zeros_like(frame)
                mask[masks[i][:, :, :] == 255] = frame[masks[i][:, :, :] == 255]
                self.register(inputCentroids[i], frame_num, mask)
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis=1).argsort()

            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                mask = np.zeros_like(frame)
                mask[masks[col][:, :, :] == 255] = frame[masks[col][:, :, :] == 255]

                thresh = cv2.threshold(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)[1]
                cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                print('len(self.video):', len(self.video))
                print('self.objects_seq[objectID]', self.objects_seq[objectID])
                if len(self.video)-1 < self.objects_seq[objectID]:
                    print('1111111')

                    while len(self.video) != self.objects_seq[objectID]:
                        self.video.append(self.bg.copy())

                    copy_bg = self.bg.copy()
                    cv2.fillConvexPoly(copy_bg, np.squeeze(cnts[0]).astype(int), 0)
                    self.video.append(cv2.add(copy_bg, mask))

                    # cv2.imshow('bg', self.bg)
                    # cv2.imshow('copy_bg', copy_bg)
                    # cv2.imshow('mask', mask)

                    # cv2.imshow('v', cv2.add(copy_bg, mask))
                    # cv2.waitKey(0)
                    self.objects_seq[objectID] += 1
                elif len(np.squeeze(cnts[0])) > 2: ##### cherrrrt
                    print('22222')
                    # print(np.squeeze(cnts[0]).astype(np.int32))
                    cv2.fillConvexPoly(self.video[self.objects_seq[objectID]], np.squeeze(cnts[0]).astype(np.int32), 0)
                    self.video[self.objects_seq[objectID]] = cv2.add(self.video[self.objects_seq[objectID]], mask)
                    # cv2.imshow('w', self.video[self.objects_seq[objectID]])
                    # cv2.imshow('bg', self.bg)
                    # cv2.imshow('mask', mask)
                    # cv2.waitKey(0)
                    self.objects_seq[objectID] += 1
                else:
                    # print(np.squeeze(cnts[0]).astype(np.int32))
                    self.objects_seq[objectID] += 1
                # self.moving_objects[objectID].set_mask(mask)

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    self.objects_seq[objectID] += 1
                    # self.video.append(self.bg.copy())
                    # self.moving_objects[objectID].set_mask(np.zeros_like(frame))
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID, frame_num)

            else:
                for col in unusedCols:
                    mask = np.zeros_like(frame)
                    mask[masks[col][:, :, :] == 255] = frame[masks[col][:, :, :] == 255]
                    self.register(inputCentroids[col], frame_num, mask)

        # return the set of trackable objects
        return self.objects

    def get_moving_objects(self):
        return self.moving_objects

    def get_video(self):
        return self.video
