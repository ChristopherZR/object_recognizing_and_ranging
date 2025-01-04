import numpy as np
import cv2
import sys
import json
import time





def sgbm_setup(params):
    left_intrinsic_matrix = np.array(params["left_intrinsic_matrix"])
    right_intrinsic_matrix = np.array(params["right_intrinsic_matrix"])
    left_distortion = np.array(params["left_distortion"])
    right_distortion = np.array(params["right_distortion"])
    rotation = np.array(params["rotation_matrix"])
    translation = np.array(params["translation_matrix"])
    width = params["camera_width"]
    height = params["camera_height"]
    half_width = int(width / 2)
    camera_size = (half_width, height)

    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        left_intrinsic_matrix, left_distortion, right_intrinsic_matrix,
        right_distortion, camera_size, rotation, translation
    )

    left_map1, left_map2 = cv2.initUndistortRectifyMap(left_intrinsic_matrix, left_distortion, R1, P1, camera_size,
                                                       cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(right_intrinsic_matrix, right_distortion, R2, P2, camera_size,
                                                         cv2.CV_16SC2)
    blockSize = 6
    img_channels = 3
    sgbm_configure = cv2.StereoSGBM_create(minDisparity=1,
                                 numDisparities=64,
                                 blockSize=blockSize,
                                 P1=8 * img_channels * blockSize * blockSize,
                                 P2=32 * img_channels * blockSize * blockSize,
                                 disp12MaxDiff=-1,
                                 preFilterCap=1,
                                 uniquenessRatio=10,
                                 speckleWindowSize=100,
                                 speckleRange=100,
                                 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    return (left_map1, left_map2, right_map1, right_map2), sgbm_configure, Q


def sgbm_run(mappings, sgbm_configure, frames, Q, boxes):
    left_map1, left_map2, right_map1, right_map2 = mappings
    frameL, frameR = frames
    frameL_gray = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    frameR_gray = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
    frameL_rectified = cv2.remap(frameL_gray, left_map1, left_map2, cv2.INTER_LINEAR)
    frameR_rectified = cv2.remap(frameR_gray, right_map1, right_map2, cv2.INTER_LINEAR)

    disparity = sgbm_configure.compute(frameL_rectified, frameR_rectified)
    threeD = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
    depth_map = threeD[:, :, 2]
    depth_map *= 16
    mean_values = []
    for box in boxes:
        x1, y1, x2, y2 = box
        result = depth_map[y1:y2, x1:x2]
        flattened_list = result.flatten().astype(int).tolist()
        flattened_list = [x for x in flattened_list if x != 160000]
        interval = 200
        sublists = split_data_by_interval(flattened_list, interval)
        final_list = find_optimal_group(sublists)
        mean_value = int(np.mean(final_list))/1000
        if mean_value < 1.5:
            mean_values.append(f"< 1.5m")
        else:
            mean_values.append(f"{mean_value:.3f}m")
    return mean_values


def split_data_by_interval(lst, interval):
    arr = np.array(lst)
    bins = np.arange(0, arr.max() + interval, interval)
    indices = np.digitize(arr, bins)
    sublists = [arr[indices == i].tolist() for i in range(1, len(bins))]
    return sublists


def find_optimal_group(sublists):
    max_indices = np.argsort([len(sublist) for sublist in sublists])[-3:][::-1]

    def sample_mean(lst):
        if len(lst) == 0:
            return float('inf')
        start = len(lst) // 4
        end = 3 * len(lst) // 4
        return np.mean(lst[start:end])

    mean_values = [sample_mean(sublists[i]) for i in max_indices]
    min_mean_index = max_indices[np.argmin(mean_values)]

    def is_adjacent(index1, index2):
        return abs(index1 - index2) == 1

    adjacent_indices = [i for i in max_indices if is_adjacent(i, min_mean_index)]

    if not adjacent_indices:
        prev_index = min_mean_index - 1 if min_mean_index > 0 else None
        next_index = min_mean_index + 1 if min_mean_index < len(sublists) - 1 else None
        final_list = sublists[min_mean_index].copy()
        if prev_index is not None:
            final_list.extend(sublists[prev_index])
        if next_index is not None:
            final_list.extend(sublists[next_index])
        return final_list

    # 使用 numpy 的方法找到 adjacent_indices 在 max_indices 中的位置
    adjacent_mean_values = [mean_values[np.where(max_indices == i)[0][0]] for i in adjacent_indices]
    max_adjacent_index = adjacent_indices[np.argmax(adjacent_mean_values)]

    target_index = min_mean_index if len(sublists[min_mean_index]) > len(
        sublists[max_adjacent_index]) else max_adjacent_index

    prev_index = target_index - 1 if target_index > 0 else None
    next_index = target_index + 1 if target_index < len(sublists) - 1 else None
    final_list = sublists[target_index].copy()
    if prev_index is not None:
        final_list.extend(sublists[prev_index])
    if next_index is not None:
        final_list.extend(sublists[next_index])
    return final_list

if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    with open("parameters.json", "r") as json_file:
        params = json.load(json_file)
    mappings, sgbm_configure, Q = sgbm_setup(params)
    frame = cv2.imread("./origin.jpg")
    frameL = frame[:, :640]
    frameR = frame[:, 640:]
    start = time.time()
    mean_value = sgbm_run(mappings, sgbm_configure, (frameL, frameR), Q, [[139, 225, 497, 555]])
    end = time.time()
    print(f"time consumed: {end - start}")
    print("mean value:", mean_value)
