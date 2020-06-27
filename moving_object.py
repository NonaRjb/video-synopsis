from collections import OrderedDict


class MovingObject:
    def __init__(self, frame_num, centroid):
        self.centroid = centroid
        self.mask = OrderedDict()
        self.nextID = 0
        self.first_frame = frame_num
        self.last_frame = None
        self.descriptor = None
        self.time = None

    def set_time(self, fps):
        if self.last_frame is not None:
            self.time = (self.last_frame - self.first_frame) / fps

    def set_last_frame(self, last_frame_num):
        self.last_frame = last_frame_num

    def set_mask(self, mask):
        self.mask[self.nextID] = mask
        self.nextID += 1

    def get_mask(self):
        return self.mask
