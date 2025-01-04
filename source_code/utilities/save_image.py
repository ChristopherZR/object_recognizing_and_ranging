import cv2
import os

# check the existence of the directory else create.
if not os.path.exists('../static/data'):
    os.mkdir('../static/data')
if not os.path.exists('../static/data/left'):
    os.makedirs('../static/data/left')
if not os.path.exists('../static/data/right'):
    os.makedirs('../static/data/right')

# initialize camera with stereo resolution 1280*480
width = 1280
height = 480
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# check
if cap.get(cv2.CAP_PROP_FRAME_WIDTH) != width:
    print(f"width set failed, currently {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    exit(1)
if cap.get(cv2.CAP_PROP_FRAME_HEIGHT) != height:
    print(f"height set failed, currently {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    exit(2)

print(f"Successfully set current resolution: {int(width)}x{int(height)}")

if not cap.isOpened():
    print("Camera not available. Check its connection and occupation")
    exit()

i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read camera frame")
        break

    cv2.imshow('Frames', frame)

    # press 's' to split and save
    if cv2.waitKey(1) & 0xFF == ord('s'):
        half_width = int(width // 2)
        left_frame = frame[:, :half_width]
        right_frame = frame[:, half_width:]

        cv2.imwrite(f'../static/data/left/{i}_left.jpg', left_frame)
        cv2.imwrite(f'../static/data/right/{i}_right.jpg', right_frame)
        print(f"left and right parts saved: {i}_left.jpg and {i}_right.jpg")

        i += 1

    # press 'q' to quit
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release
cap.release()
cv2.destroyAllWindows()
