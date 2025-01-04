import cv2
import numpy as np
from dataclasses import dataclass



@dataclass
class ProcessorArgs:
    pass


class Processor:
    def __init__(self):
        pass



    @staticmethod
    def preprocess(frame):
        """reshape the frame to 640*640, fitting yolo's requirement"""
        shape = frame.shape
        height = shape[0]
        width = shape[1]
        rate = min(640 / height, 640 / width)
        new_height = round(rate * height)
        new_width = round(rate * width)
        diff_width = int(640 - new_width)
        diff_height = int(640 - new_height)
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        top, left, bottom, right = (round(diff_height / 2), round(diff_width / 2),
                                    round(diff_height / 2), round(diff_width / 2))
        frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                   value=(114, 114, 114))  # add border
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.expand_dims(frame, 0)
        frame = frame.copy()  # 解决负步长问题
        return frame

    def postprocess(self, input_data, OBJ_THRESH, NMS_THRESH):
        """ input: raw output data from the model
            output: boxes and their classes and their scores"""
        boxes, scores, classes_conf = [], [], []
        defualt_branch = 3
        pair_per_branch = len(input_data) // defualt_branch
        # Python 忽略 score_sum 输出
        for i in range(defualt_branch):
            boxes.append(self._box_process(input_data[pair_per_branch * i]))
            classes_conf.append(input_data[pair_per_branch * i + 1])
            scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

        def sp_flatten(_in):
            ch = _in.shape[1]
            _in = _in.transpose(0, 2, 3, 1)
            return _in.reshape(-1, ch)

        boxes = [sp_flatten(_v) for _v in boxes]
        classes_conf = [sp_flatten(_v) for _v in classes_conf]
        scores = [sp_flatten(_v) for _v in scores]

        boxes = np.concatenate(boxes)
        classes_conf = np.concatenate(classes_conf)
        scores = np.concatenate(scores)
        boxes = boxes.astype(int)
        # filter according to threshold
        boxes, classes, scores = self._filter_boxes(boxes, scores, classes_conf, OBJ_THRESH)

        # nms
        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            keep = self._nms_boxes(b, s, NMS_THRESH)

            if len(keep) != 0:
                nboxes.append(b[keep])
                nclasses.append(c[keep])
                nscores.append(s[keep])

        if not nclasses and not nscores:
            return None, None, None

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)
        return boxes, classes, scores

    @staticmethod
    def _filter_boxes(boxes, box_confidences, box_class_probs, OBJ_THRESH):
        """Filter the boxes according to the object thresholds and give the scores to the new boxes"""
        # flatten box_confidence (8400, 1) to (8400,), one dimension
        box_confidences = box_confidences.reshape(-1)
        # get the largest scores "box_class_probs" along with axis(1)
        class_max_score = np.max(box_class_probs,axis=-1)
        # get index of the largest scores "box_class_probs" along with axis(1)
        classes = np.argmax(box_class_probs,axis=-1)
        # get indexes of boxes that pass the threshold
        indexes = np.where(class_max_score * box_confidences >= OBJ_THRESH)
        # multiply confidences by the boxes array to get scores
        scores = (class_max_score * box_confidences)[indexes]
        boxes = boxes[indexes]  # boxes that pass the threshold
        classes = classes[indexes]  # indexes-to-classes of boxes that pass the threshold


        return boxes, classes, scores

    @staticmethod
    def _nms_boxes(boxes, scores, NMS_THRESH):
        """Filter the duplicated boxes on the same objects while remaining the ones with the biggest IOU"""
        x = boxes[:, 0]  # get all lines in column 0 which are the left-up x values
        y = boxes[:, 1]  # get all lines in column 1 which are the left-up y values
        w = boxes[:, 2] - boxes[:, 0]  # get the widths of the boxes
        h = boxes[:, 3] - boxes[:, 1]  # get the heights of the boxes
        areas = w * h  # area equals to width times height
        order = scores.argsort()[::-1]  # get the decreasing order of the scores
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])  # left-up x value of intersection area
            yy1 = np.maximum(y[i], y[order[1:]])  # left-up y value of intersection area
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])  # right-bottom x value of intersection area
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])  # right-bottom y value of intersection area

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)  # non-negative (minimum 0)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1  # intersection area

            ious = inter / (areas[i] + areas[order[1:]] - inter)  # calculate IOUs: intersection over union
            inds = np.where(ious <= NMS_THRESH)[0]  # get indexes of satisfied IOUs
            order = order[inds + 1]  # jump to ignore the unsatisfied ones
        keep = np.array(keep)
        return keep

    @staticmethod
    def _softmax(x, axis=None):
        x = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=axis, keepdims=True)

    # instance method
    def _dfl(self, position):
        """not studied yet"""
        # Distribution Focal Loss (DFL)
        n, c, h, w = position.shape
        p_num = 4
        mc = c // p_num
        y = position.reshape(n, p_num, mc, h, w)
        y = self._softmax(y, 2)
        acc_metrix = np.array(range(mc), dtype=float).reshape(1, 1, mc, 1, 1)
        y = (y * acc_metrix).sum(2)
        return y

    # instance method
    def _box_process(self, position):
        """not studied yet"""
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array([640 // grid_h, 640 // grid_w]).reshape(1, 2, 1, 1)

        position = self._dfl(position)
        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

        return xyxy

























