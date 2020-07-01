from collections import OrderedDict


class MovingObject:
    def __init__(self, frame_num, centroid):
        self.centroid = [centroid]
        self.mask = OrderedDict()
        self.nextID = 0
        self.first_frame = frame_num
        self.last_frame = None
        self.descriptor = None
        self.time = None

    def set_time(self, fps):
        if self.last_frame is not None:
            self.time = (self.last_frame - self.first_frame + 1) / fps

    def set_last_frame(self, last_frame_num):
        self.last_frame = last_frame_num

    def set_mask(self, mask):
        self.mask[self.nextID] = mask
        self.nextID += 1

    def get_mask(self):
        return self.mask

    def get_last_frame(self):
        return self.last_frame

    def get_first_frame(self):
        return self.first_frame

    def get_time(self):
        return self.time

    def set_centroid(self, centroid):
        self.centroid.append(centroid)

    def copy_centroid(self):
        last_centroid = self.centroid[-1]
        self.centroid.append(last_centroid)

    def get_centroid(self, frame_id):
        return self.centroid[frame_id]

    def get_len_centroid(self):
        return len(self.centroid)
