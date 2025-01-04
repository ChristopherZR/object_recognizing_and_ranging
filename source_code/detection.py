from dataclasses import dataclass
from rknnlite.api import RKNNLite
from processor import Processor
import cv2
import numpy as np


@dataclass
class YoloArgs:
    OBJ_THRESH: float = 0.8
    NMS_THRESH: float = 0.8
    origin_frame = None
    model_path: str = "static/model/yolov10s.rknn"


class YOLO:
    def __init__(self, args: YoloArgs):
        self.args = args
        self.model = self._model_init()
        self.processor = Processor()
        self.position: dict = {}
        self.boxes, self.classes, self.scores = None, None, None

    def run(self, frame):
        original_frame, boxes, classes, scores = self._get_boxes(frame)
        original_frame = np.squeeze(original_frame)
        if boxes is not None:
            # processed_frame = self.draw_detections(boxes, scores, self.args.origin_frame)
            return boxes, scores, original_frame
        else:
            return None, None, None


    def _model_init(self):
        model = RKNNLite(verbose=False)
        model.load_rknn(self.args.model_path)
        model.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        return model

    def _get_boxes(self, frame):
        self.args.origin_frame = frame
        frame = self.processor.preprocess(frame)
        outputs = self.model.inference(inputs=[frame], data_format='nhwc')
        boxes, classes, scores = self.processor.postprocess(outputs, self.args.OBJ_THRESH, self.args.NMS_THRESH)
        if boxes is not None:
            boxes = boxes[classes == 0]
            return self.args.origin_frame, boxes, classes, scores
        else:
            return None, None, None, None

    @staticmethod
    def draw_detections(boxes: dict, scores, origin_frame, distance):
        import random
        """
        Draws bounding boxes and scores on the input image for a single class of detections.

        Args:
            image (numpy.ndarray): The input image to draw detections on, shape [1, H, W, 3].
            boxes (numpy.ndarray): Array of bounding boxes with shape (n, 4).
                                   Each box is [x1, y1, x2, y2] in pixel coordinates.
            scores (numpy.ndarray): Array of detection scores with shape (n,).
            color (tuple): Color for drawing boxes and text (default is green).

        Returns:
            numpy.ndarray: The image with drawn detections, shape [H, W, 3].
        """

        def to_origin(coord, origin_frame):
            origin_shape = origin_frame.shape
            rate = 640 / origin_shape[1]
            valid_height = origin_shape[0] * rate
            padding_height = int((640 - valid_height) / 2)
            top, bottom, left, right = coord
            top = (top - padding_height) / rate
            bottom = (bottom - padding_height) / rate
            left = left / rate
            right = right / rate

            return top, bottom, left, right

        for idx, box in enumerate(boxes):
            if True:
                color_list = [(73, 71, 188), (183, 144, 123), (233, 54, 209), (233, 96, 214), (104, 191, 82),
                              (227, 150, 79), (195, 129, 71), (52, 90, 247), (167, 125, 82), (175, 78, 133),
                              (113, 69, 184), (253, 174, 209), (105, 145, 55), (54, 85, 233), (213, 99, 120),
                              (55, 148, 237), (53, 77, 227), (137, 174, 150), (196, 155, 120), (206, 136, 222),
                              (153, 136, 155), (121, 156, 194), (100, 121, 184), (190, 62, 248), (87, 75, 186),
                              (181, 64, 65), (236, 146, 164), (224, 115, 247), (255, 189, 150), (196, 52, 87),
                              (238, 68, 250), (98, 50, 164), (75, 83, 230), (97, 77, 189), (215, 200, 204),
                              (56, 96, 106), (168, 147, 235), (145, 99, 216), (188, 75, 88), (193, 50, 166),
                              (216, 165, 87), (57, 120, 114), (132, 186, 221), (130, 109, 140), (226, 152, 102),
                              (84, 112, 102), (196, 58, 154), (104, 151, 184), (195, 124, 123), (119, 127, 90),
                              (102, 108, 182), (61, 59, 119), (100, 185, 93), (230, 198, 214), (254, 165, 135),
                              (178, 153, 120), (207, 127, 93), (222, 58, 108), (121, 137, 60), (203, 165, 245),
                              (215, 161, 153), (124, 164, 178), (185, 78, 119), (214, 172, 95), (72, 114, 234),
                              (238, 97, 220), (211, 78, 234), (227, 176, 251), (128, 72, 239), (115, 65, 54),
                              (76, 109, 209), (200, 171, 94), (209, 153, 71), (74, 140, 164), (207, 113, 56),
                              (164, 135, 172), (158, 195, 96), (225, 66, 72), (75, 85, 171), (191, 59, 58),
                              (87, 134, 226), (214, 111, 183), (90, 192, 163), (54, 118, 147), (208, 184, 142),
                              (181, 133, 205), (61, 56, 225), (105, 128, 211), (140, 164, 65), (156, 171, 121),
                              (232, 110, 123), (243, 188, 141), (198, 186, 165), (171, 150, 249), (244, 82, 54),
                              (122, 59, 61), (215, 160, 72), (112, 141, 161), (159, 172, 73), (228, 115, 85)]
                color = color_list[idx]
                left, top, right, bottom = map(int, box)  # apply int() to all elements in 'box'
                top, bottom, left, right = to_origin((top, bottom, left, right), origin_frame)
                top, bottom, left, right = map(int, (top, bottom, left, right))
                cv2.rectangle(origin_frame, (left, top), (right, bottom), color,
                              2)  # image, left-up, right-bottom, color, thickness
                if idx < len(distance):
                    label = f"person{idx}: {round(float(scores[idx]), 2)}, d: {distance[idx]}"
                else:
                    label = f"person{idx}: {round(float(scores[idx]), 2)}, d: N/A m"
                # Calculate the dimensions of the label text
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                # Calculate the position of the label text
                label_x = left
                label_y = top - 20

                # Draw a filled rectangle as the background for the label text
                cv2.rectangle(origin_frame, (label_x, label_y - label_height),
                              (label_x + label_width, label_y + label_height),
                              color, cv2.FILLED)

                # Draw the label text on the image
                cv2.putText(origin_frame, label, (label_x, label_y + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        processed_frame = origin_frame
        return processed_frame



if __name__ == "__main__":
    detect = YOLO(YoloArgs())
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width, height)
    for i in range(3):
        success, frame = camera.read()
    left_frame = frame[:, :640]
    right_frame = frame[:, 640:]
    cv2.imwrite("origin.jpg", frame)
    if not success:
        print("camera is unavailable")
        exit(1)
    result, boxes = detect.run(right_frame)
    print(boxes)





