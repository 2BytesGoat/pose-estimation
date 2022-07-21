# %%
import numpy as np
import mediapipe as mp
mp_keypoints = mp.solutions.pose.PoseLandmark

data_path = 'data/imgs/img0.txt'
with open(data_path, 'r') as f:
    data = np.array(eval(f.read()))

mp_mapping = {
        'lefthip': mp_keypoints.LEFT_HIP.value,
        'leftknee': mp_keypoints.LEFT_KNEE.value,
        'leftfoot': mp_keypoints.LEFT_HEEL.value,

        'righthip': mp_keypoints.RIGHT_HIP.value,
        'rightknee': mp_keypoints.RIGHT_KNEE.value,
        'rightfoot': mp_keypoints.RIGHT_HEEL.value, 

        'leftshoulder': mp_keypoints.LEFT_SHOULDER.value,
        'leftelbow': mp_keypoints.LEFT_ELBOW.value,
        'leftwrist': mp_keypoints.LEFT_WRIST.value,

        'rightshoulder': mp_keypoints.RIGHT_SHOULDER.value,
        'rightelbow': mp_keypoints.RIGHT_ELBOW.value,
        'rightwrist': mp_keypoints.RIGHT_WRIST.value
    }

kpts = {
    key: data[value] for key, value in mp_mapping.items()
}
# %%
with open('frame_0.txt', 'r') as f:
    kpts = eval(f.read())
kpts = {key: np.array(value) for key, value in kpts.items()}
# %%
from keypoints import KeypointRotations

calculator = KeypointRotations()
new_kpts = calculator.add_neck_and_hip_keypoints(kpts)
new_kpts = calculator.center_keypoints(new_kpts, 'hips')
angles = calculator.calculate_keypoint_angles(new_kpts)
# %%
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

bone_lengths = calculator.get_bone_lengths(new_kpts)
normalization = bone_lengths['neck']
base_skeleton = calculator.get_base_skeleton(bone_lengths, normalization)

for joint in new_kpts:
    r1, r2 = calculator.reconstruct_joint_kpts_from_angles(
        joint, angles, base_skeleton, new_kpts['hips'], normalization
    )
    plt.plot(xs = [r1[0], r2[0]], ys = [r1[1], r2[1]], zs = [r1[2], r2[2]], color = 'red')

plt.show()
# %%
