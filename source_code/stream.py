from flask import Flask, Response
import cv2
import json
import copy

import sgbm
from detection import YOLO, YoloArgs



yolo = YOLO(YoloArgs())

app = Flask(__name__)
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"当前分辨率: {width}x{height}")


def get_frames():
    with open("parameters.json") as json_file:
        params = json.load(json_file)
    mappings, sgbm_configure, Q = sgbm.sgbm_setup(params)
    i = 0
    mean_values = None # initialize
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            left_frame = frame[:, :640]
            right_frame = frame[:, 640:]
            pre_process_frame = copy.deepcopy(right_frame)
            # obtain the yolo image
            boxes, scores, original_frame = yolo.run(pre_process_frame)
            if boxes is not None:
                # obtain the distance image
                if i & 5 == 0 or mean_values is None: # reduce the FPS of distance measuring
                    mean_values = sgbm.sgbm_run(mappings, sgbm_configure, (left_frame, right_frame), Q, boxes)
                yolo_frame = yolo.draw_detections(boxes, scores, original_frame, mean_values)
            else:
                yolo_frame = right_frame[:]
            _, buffer = cv2.imencode('.jpg', yolo_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        i += 1


@app.route('/video_feed')
def video_feed():
    return Response(get_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # Replace with your desired port