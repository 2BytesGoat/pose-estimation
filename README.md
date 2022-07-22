# Godot Pose Estimation 
The goal of this project is to move a 3D character from Godot based on user movement.

## Overview
1. Pose and depth information are extracted from a frame.
2. Angles between pose keypoints are calculated.
3. Angle information is transmitted to Godot via UDP.
4. We apply the rotation angles on the 3D character to mimic the pose extracted from the frame. 

## Special thanks to
* https://github.com/TemugeB - for the [joint-angles-calculate](https://github.com/TemugeB/joint_angles_calculate) starter code

## Bibliography
1. https://ai.googleblog.com/2020/08/on-device-real-time-body-pose-tracking.html
2. https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html
3. https://www.andre-gaschler.com/rotationconverter/

## ToDos
* [x] debug `joint-angles-calculate` refactor to see why we have bad outpus
* [x] add tests for `keypoint-rotations` 
* [x] do skeleton reconstruction based on angles
* [x] check what is the diference between `pose_landmarks` and `pose_world_landmarks`
* [x] see whether angle order coincide with TB's approach 
* [ ] see whether angles can be enforced on different skeleton shape
* [ ] document the calculation of joint angles procedure
* [ ] make project readme prettier
