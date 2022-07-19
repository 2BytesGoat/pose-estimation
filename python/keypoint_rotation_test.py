import numpy as np
from keypoints.keypoint_rotation import KeypointRotations

kpts_sample = {
    'lefthip': np.array([0.667934  , 8.87940658, 9.96319157]), 
    'leftknee': np.array([-3.34578655,  8.96091348,  9.73515891]), 
    'leftfoot': np.array([-5.92501355, 11.27615927,  7.50971397]), 
    'righthip': np.array([ 0.75793423, 10.31887436, 11.43529256]), 
    'rightknee': np.array([-3.38370222,  9.46210077, 14.07380719]), 
    'rightfoot': np.array([-7.25593058, 10.16061439, 13.68205105]), 
    'leftshoulder': np.array([6.37315284, 8.42415094, 8.77096848]), 
    'leftelbow': np.array([3.26082629, 8.24708268, 8.22167763]), 
    'leftwrist': np.array([0.54430702, 6.99003009, 8.91145991]), 
    'rightshoulder': np.array([ 6.69674782, 10.98142669, 11.50536544]), 
    'rightelbow': np.array([ 3.55587357, 11.87956765, 12.13774564]), 
    'rightwrist': np.array([ 0.48589837, 11.52834985, 13.55304308]), 
}

def test_add_neck_and_hip_keypoints():
    target_neck = np.array([6.535, 9.703, 10.138])
    target_hips = np.array([0.713, 9.599, 10.699])
    
    kp_obj = KeypointRotations()
    new_kpts = kp_obj.add_neck_and_hip_keypoints(kpts_sample)

    for joint, target_value in zip(
        ['neck', 'hips'], 
        [target_neck, target_hips]):
        
        output_value = np.around(new_kpts[joint], 3)
        assert(target_value == output_value).all(), \
            f'Incorrect {joint}-keypoint, expected {target_value} but got {output_value}'

def test_center_keypoints():
    target_hips = np.array([0., 0., 0.])
    target_leftknee = np.array([-4.059, -0.638, -0.964])
    target_rightelbow = np.array([2.843, 2.28, 1.439])

    kp_obj = KeypointRotations()
    new_kpts = kp_obj.add_neck_and_hip_keypoints(kpts_sample)
    new_kpts = kp_obj.center_keypoints(new_kpts, 'hips')

    for joint, target_value in zip(
        ['hips', 'leftknee', 'rightelbow'], 
        [target_hips, target_leftknee, target_rightelbow]):
        
        output_value = np.around(new_kpts[joint], 3)
        assert(target_value == output_value).all(), \
            f'Incorrect {joint}-keypoint, expected {target_value} but got {output_value}'

def test_get_joint_rotations():
    target_lefthip_rot = np.array([0.083, 0.134, -0.006])

    kp_obj = KeypointRotations()
    new_kpts = kp_obj.add_neck_and_hip_keypoints(kpts_sample)
    new_kpts = kp_obj.center_keypoints(new_kpts, 'hips')

    joint = 'leftknee'
    kpts_rotations = {'hips': np.array([0., 0. ,0.])}
    connected_joints = kp_obj.kpts_hierarchy[joint]
    parent = connected_joints[0]

    output_lefthip_rot = kp_obj.get_joint_rotations(
        new_kpts[joint], 
        kp_obj.kpts_offsets[joint], 
        new_kpts[parent], 
        connected_joints, 
        kpts_rotations)

    assert(target_lefthip_rot == output_lefthip_rot).all(), \
        f'Incorrect {joint}-keypoint, expected {target_lefthip_rot} but got {output_lefthip_rot}'

if __name__ == '__main__':
    test_add_neck_and_hip_keypoints()
    test_center_keypoints()
    test_get_joint_rotations()
