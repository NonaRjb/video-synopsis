from collections import OrderedDict
import numpy as np
from scipy.spatial import distance as dist


def compute_correspondence_mat(D):
    C = np.zeros_like(D).astype(int)
    if np.all(D == 1024):
        return C
    cols = D.argmin(axis=1)
    rows = D.argmin(axis=0)
    for i in range(len(cols)):
        C[i, cols[i]] += 1
    for j in range(len(rows)):
        C[rows[j], j] += 1
    return C


def compute_correspondence_mat_res(D):
    C = np.zeros_like(D).astype(int)
    cols = D.argmin(axis=1)
    rows = D.argmin(axis=0)
    for i in range(len(cols)):
        C[i, cols[i]] = 1
    for j in range(len(rows)):
        C[rows[j], j] = 1
    return C


class Tracker:
    def __init__(self, thresh, maxDisappeared=20):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.objects_hist = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.thresh = thresh

    def register(self, centroid, rgb_hist=None):
        self.objects[self.nextObjectID] = centroid
        self.objects_hist[self.nextObjectID] = rgb_hist
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        # del self.objects_hist[objectID]
        del self.disappeared[objectID]

    def update(self, rects, frame=None):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            D[D > self.thresh] = 1024

            C = compute_correspondence_mat(D)

            usedRows = set()
            usedCols = set()

            while C[C == 2].size is not 0 and ~np.all(D == 1024):
                for row, col in np.argwhere(C == 2):
                    objectID = objectIDs[row]
                    self.objects[objectID] = inputCentroids[col]
                    self.disappeared[objectID] = 0
                    D[:, col] = 1024
                    D[row, :] = 1024
                    usedRows.add(row)
                    usedCols.add(col)
                C = compute_correspondence_mat(D)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if len(unusedRows)>0:
                remained_tracks = []
                remained_centroids = []
                for r in range(len(unusedRows)):
                    remained_tracks.extend([objectIDs[r]])
                    remained_centroids.extend([objectCentroids[r]])

                D_res = dist.cdist(np.array(remained_centroids), inputCentroids)
                D_res[D_res > self.thresh] = 1024

                C_res = compute_correspondence_mat_res(D_res)
                print('C_res:')
                print(C_res)

            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            for col in unusedCols:
                self.register(inputCentroids[col])

        return self.objects

