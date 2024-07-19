from abc import ABC
import numpy as np
import cv2


class BaseTransform(ABC):
    def __init__(self):
        pass

    def __call__(self, x):
        return x


class DivideByScalar(BaseTransform):
    def __init__(self, scalar, axis=None, channels=None):
        super(DivideByScalar, self).__init__()
        self.scalar = scalar
        self.axis = axis
        self.channels = channels

    def __call__(self, x):
        if self.axis and self.channels:
            # axis is dimension to index into
            # channels is list of channels on that axis to divide
            idx = [slice(None)] * x.ndim
            for c in self.channels:
                idx[self.axis] = c
                x[tuple(idx)] /= self.scalar
        else:
            x /= self.scalar

        return x


class Reshape(BaseTransform):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def __call__(self, x):
        return x.reshape(self.shape)
    

class MoveAxis(BaseTransform):
    def __init__(self, orig_axis, new_axis):
        super(MoveAxis, self).__init__()
        self.orig_axis = orig_axis
        self.new_axis = new_axis

    def __call__(self, x):
        return np.moveaxis(x, self.orig_axis, self.new_axis)


class Difference(BaseTransform):
    # takes diff of first axis and concatenates as new first axis
    def __init__(self):
        super(Difference, self).__init__()

    def __call__(self, x):
        diff = np.diff(x, axis=0)
        return np.stack([x[1:], diff])
    

class DifferenceOpticalFlow(BaseTransform):
    def __init__(self):
        super(DifferenceOpticalFlow, self).__init__()

    def __call__(self, x):
        grey = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in x]
        flows = [cv2.calcOpticalFlowFarneback(grey[i], grey[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0) for i in range(len(grey) - 1)]
        flows = np.array(flows)
        return np.concatenate([x[1:], flows], axis=-1)
    

class StackAxis(BaseTransform):
    def __init__(self, orig_axis, merge_axis):
        super(StackAxis, self).__init__()
        self.orig_axis = orig_axis
        self.merge_axis = merge_axis

    def __call__(self, x):
        # if orig shape is (2, 2, 3, 360, 640)
        # and orig_axis is 0, merge_axis is 2
        # new shape is (2, 6, 360, 640)
        return np.concatenate(np.split(x, x.shape[self.orig_axis], axis=self.orig_axis), axis=self.merge_axis)[0]