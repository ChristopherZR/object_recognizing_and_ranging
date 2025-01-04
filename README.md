# RKNN-Based Stereo Depth and Object Detection System

This repository provides a complete implementation of a stereo depth measurement and object detection system developed for the RK3588S platform. The system utilizes the YOLO model for object detection and stereo camera with SGBM (Semi-Global Block Matching) for stereo depth computation. The system is encapsulated in a Flask application for real-time video streaming, allowing the visualization object detection results and their distances through web interface.

## Platform
- Orange Pi RK3588s, 4GM RAM, 32G EMMC (6 Tops NPU)
- 汇博视捷 stereo camera, 3mm focal length, 120mm baseline, 80° no distortion, resolution set 1280*480@30fps
- Ubuntu 22.04.5 LTS aarch64
- Python 3.12.8

## File Structure
```
object_recognizing_and_ranging/
├── resources/
│   ├── additional_resources.txt
│   └── requirement.txt
├── source_code/
│   ├── static/
│   │   ├── data/
│   │   │   ├── left/
│   │   │   └── right/
│   │   └── model/
│   │       ├── yolov8m.rknn
│   │       └── yolov10s.rknn
│   ├── utilities/
│   │   ├── calibration.py
│   │   └── save_image.py
│   ├── detection.py
│   ├── parameters.json
│   ├── processor.py
│   ├── sgbm.py
│   └── stream.py
```
## Deployment
### 1. Install Dependencies:
`pip3 install -r ./resources/requirements.txt`
### 2. Configure the Stereo Camera
Update ./source_code/parameters.json with your camera's calibration parameters. [camera calibration guideline reference video (Mandarin Chinese)](https://www.bilibili.com/video/BV1GP41157Ti/?spm_id_from=333.337.search-card.all.click) (落叶_随峰, 2022). 
#### Summary:
  1. Print chessboard image on paper. Note down the side length of the squares
  2. Set the stereo camera to wanted resolution. You may NOT change this during usage without another calibration. 1280x480 is recommanded.
  3. Build directory: Ubuntu/Mac: `mkdir data && cd data && mkdir right left`
```
     data
     |──left
     |──right
```
  5. Run [./source_code/utilities/save_image.py](./source_code/utilities/save_image.py). to take pictures of the chessboard with the camera. Make sure the chessboard completely presents in every photo taken and make chessboard's angle as diverse as possible.  
  6. Calibration:
     1. MatLab:
        1. Import image directory into MatLab --> Apps --> Stereo Camera Calibration --> Add Images. Fill in chessboard size.
        2. Click "Calibrate". Once finished, click reprojection errors below the image and delete the ones with high errors. Keep the Overall Mean Error in 0.25 pixels, the lower the better.
        3. Click Export Camera Parameters. Return to MatLab main window --> workspace --> stereoparams
        4. Update ./source_code/parameter.json:
           ```
           "camera_width": the width you set for taking chessboard pictures, (Int)
           "camera_height": the height you set for taking chessboard pictures, (Int)
           "left_intrinsic_matrix" = stereoParams/CameraParameters1/Intrinsics/K (3*3 matrix, i.e., a 3D list)
           "right_intrinsic_matrix" = stereoParams/CameraParameters2/Intrinsics/K (3*3 matrix)
           "left_distortion" = stereoParams/CameraParameters1/Intrinsics/RadialDistortion + Tangential Distortion (1*4 matrix)
           "right_distortion" = stereoParams/CameraParameters2/Intrinsics/RadialDistortion + Tangential Distortion (1*4 matrix)
           "rotation_matrix" = stereoParams/PoseCamera2/R (3*3 matrix)
           "translation_matrix" = stereoParams/Posecamera2/Translation (1*3 matriox)
           ```
     2. OpenCV:
        Run [./source_code/utilities/clibration.py](./source_code/utilities/clibration.py). The result will be automatically saved as a json file.

### 3. Run [./source_code/stream.py](./source_code/stream.py)

## References

- [RKNN Platform and Toolkit Documentation](https://github.com/airockchip/rknn-toolkit2/tree/master/doc)

- 笨小蛙. (2024). YOLOv8目标检测部署RK3588全过程，附代码pt-＞onnx-＞rknn，附【详细代码】. Csdn.net. https://blog.csdn.net/GREEN_cq/article/details/141607095  

- 基于python的双目标定_python双目标定-CSDN博客. (2023). Csdn.net. https://blog.csdn.net/weixin_43788282/article/details/131166699  

- King, L. (2023). YOLOv5-6.1从训练到部署（三）：模型在CPU上部署_yolo模型部署-CSDN博客. Csdn.net. https://blog.csdn.net/TANTANWANG/article/details/134959739  

- 落叶_随峰. (2022). https://www.bilibili.com/video/BV1GP41157Ti/. In Bilibili.com. https://doi.org/10754352/810754352-1-30032.m4s  

- Orange Pi - Orangepi. (2025). Orangepi.org. http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/service-and-support/Orange-Pi-5-Pro.html

## License

This project is licensed under the terms of the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).  
See the [LICENSE](./LICENSE) file for details.

### Key Points:
- You are free to use, modify, and distribute this software, but **any modifications or derived works must also be licensed under GPL-3.0**.
- If you distribute this software (modified or unmodified), you **must include the source code** or provide a written offer to supply the source code.
- This software is provided "as is," without warranty of any kind, express or implied.

For more information, visit the [GNU GPL-3.0 official website](https://www.gnu.org/licenses/gpl-3.0.en.html).



     
      



