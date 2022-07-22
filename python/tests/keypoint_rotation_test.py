import numpy as np
from keypoints import KeypointRotations

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

def test_rotate_pose_up():
    target_leftfoot = np.array([-11.276, -5.925, 7.51 ])
    target_rightshoulder = np.array([-10.981, 6.697, 11.505])

    kp_obj = KeypointRotations()
    new_kpts = kp_obj.rotate_pose(kpts_sample, 'z')

    for joint, target_value in zip(
        ['leftfoot', 'rightshoulder'], 
        [target_leftfoot, target_rightshoulder]):

        output_value = np.around(new_kpts[joint], 3)
        assert(target_value == output_value).all(), \
            f'Incorrect {joint}-keypoint, expected {target_value} but got {output_value}'

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

def test_calculate_root_rotation():
    root_joint = 'hips'
    target_root_rot = np.array([-1.553, -0.096, 0.8])

    kp_obj = KeypointRotations()
    new_kpts = kp_obj.add_neck_and_hip_keypoints(kpts_sample)
    new_kpts = kp_obj.center_keypoints(new_kpts, root_joint)

    output_root_rot = kp_obj._calculate_root_rotation(
        new_kpts[root_joint], new_kpts['lefthip'], new_kpts['neck'])
    output_root_rot = np.around(output_root_rot, 3)

    assert(target_root_rot == output_root_rot).all(), \
        f'Incorrect {root_joint}-keypoint, expected {target_root_rot} but got {output_root_rot}'

def test_get_bone_lengths():
    target_bone_lengths = {
        'lefthip': 1.03, 'leftknee': 4.021, 'leftfoot': 4.119, 
        'righthip': 1.03, 'rightknee': 4.985, 'rightfoot': 3.954, 
        'leftshoulder': 1.879, 'leftelbow': 3.165, 'leftwrist': 3.072, 
        'rightshoulder': 1.879, 'rightelbow': 3.327, 'rightwrist': 3.399, 
        'neck': 5.85
    }

    kp_obj = KeypointRotations()
    new_kpts = kp_obj.add_neck_and_hip_keypoints(kpts_sample)
    output_bone_lengths = kp_obj.get_bone_lengths(new_kpts)

    for joint in target_bone_lengths:
        target_length = target_bone_lengths[joint]
        output_length = round(output_bone_lengths[joint], 3)
        assert target_length == output_length, \
            f'Expected {joint} length to be {target_length} but got {output_length}'

def test_get_rotation_chain():
    target_rotation = np.array(
        [[ 0.02646302,  0.99818489, -0.05409836],
         [-0.6982472,  -0.02027021, -0.71556968],
         [-0.71536743,  0.05671017,  0.6964434 ]]
    )
    kp_obj = KeypointRotations()

    hierarchy = ['lefthip', 'hips']
    kpts_rotations = {
        'hips': np.array([-1.55299538, -0.09598362,  0.79955135]),
        'lefthip': np.array([ 0.08336282,  0.13366153, -0.00558273]),
        'leftknee': np.array([-0.0337819 ,  0.83998899,  0.01508699]),
    }

    output_rotation = kp_obj.get_rotation_chain(hierarchy, kpts_rotations)
    
    target_rotation = np.around(target_rotation, 3)
    output_rotation = np.around(output_rotation, 3)

    print('Output rotation: ', output_rotation)

    assert (target_rotation == output_rotation).all(), \
        f'Expected lefthip rotation to be {target_rotation} but got {output_rotation}'

def test_get_rotation_chain_inverse():
    target_rotations = {
        'lefthip': np.array([[-0.056, -0.698, -0.714],[ 0.995,  0.018, -0.096],[ 0.08 , -0.716,  0.693]]), 
        'righthip': np.array([[-0.056, -0.698, -0.714],[ 0.995,  0.018, -0.096],[ 0.08 , -0.716,  0.693]]), 
        'neck': np.array([[-0.056, -0.698, -0.714],[ 0.995,  0.018, -0.096],[ 0.08 , -0.716,  0.693]]), 
        'leftknee': np.array([[ 0.026, -0.698, -0.715],[ 0.998, -0.021 ,  0.057],[-0.054, -0.716,  0.696]]), 
        'rightknee': np.array([[-0.261, -0.719, -0.644],[ 0.831,  0.172, -0.529],[ 0.491, -0.673,  0.553]]), 
        'leftshoulder': np.array([[-0.086, -0.68, -0.728],[ 0.993, -0.002, -0.116],[ 0.077, -0.733,  0.676]]), 
        'rightshoulder': np.array([[-0.086, -0.68, -0.728],[ 0.993, -0.002, -0.116],[ 0.077, -0.733,  0.676]]), 
        'leftelbow': np.array([[-0.983, -0.056, -0.173],[ 0.174, -0.565, -0.806],[-0.053, -0.823,  0.565]]), 
        'rightelbow': np.array([[ 0.944, -0.27 , -0.19 ],[ 0.33,  0.733,  0.594],[-0.021, -0.624,  0.781]])
    }

    kpts_angles = {
        'hips': np.array([-1.553, -0.096,  0.8  ]), 
        'lefthip': np.array([ 0.083,  0.134, -0.006]), 
        'righthip': np.array([-0.235, -0.438, -0.053]), 
        'neck': np.array([-0.028, -0.   ,  0.025]), 
        'leftknee': np.array([-0.034,  0.84 ,  0.015]), 
        'rightknee': np.array([ 0.258,  0.714, -0.097]), 
        'leftshoulder': np.array([-1.298, -0.117,  0.154]), 
        'rightshoulder': np.array([ 1.309, -0.111, -0.144]), 
        'leftelbow': np.array([-0.104,  0.029, -0.536]), 
        'rightelbow': np.array([0.126, 0.027, 0.421])
    }

    kp_obj = KeypointRotations()
    new_kpts = kp_obj.add_neck_and_hip_keypoints(kpts_sample)
    new_kpts = kp_obj.center_keypoints(new_kpts, 'hips')

    output_rotations = {}
    for depth in range(2, kp_obj.max_connected_joints+1):    
        for joint, connected_joints in kp_obj.kpts_hierarchy.items():
            if len(connected_joints) != depth: 
                continue
            parent = connected_joints[0]
            R = kp_obj.get_rotation_chain(connected_joints[1:], kpts_angles, inverse=True)
            output_rotations[parent] = np.around(R, 3)
    
    for joint in target_rotations:
        target_angle = target_rotations[joint]
        output_angle = output_rotations[joint]
        print('-'*5 + joint + '-'*5)
        print('target:', target_angle)
        print('output:', output_angle)
        assert (target_angle == output_angle).all(), \
            f'Expected {joint} rotation to be {target_angle} but got {output_angle}'

def test_calculate_keypoint_angles():
    target_angles = {
        'hips': np.array([-1.553, -0.096,  0.8  ]), 
        'lefthip': np.array([ 0.083,  0.134, -0.006]), 
        'righthip': np.array([-0.235, -0.438, -0.053]), 
        'neck': np.array([-0.028, -0.   ,  0.025]), 
        'leftknee': np.array([-0.034,  0.84 ,  0.015]), 
        'rightknee': np.array([ 0.258,  0.714, -0.097]), 
        'leftshoulder': np.array([-1.298, -0.117,  0.154]), 
        'rightshoulder': np.array([ 1.309, -0.111, -0.144]), 
        'leftelbow': np.array([-0.104,  0.029, -0.536]), 
        'rightelbow': np.array([0.126, 0.027, 0.421])
    }

    kp_obj = KeypointRotations()
    new_kpts = kp_obj.add_neck_and_hip_keypoints(kpts_sample)
    new_kpts = kp_obj.center_keypoints(new_kpts, 'hips')

    output_angles = kp_obj.calculate_keypoint_angles(new_kpts)
    
    for joint in target_angles:
        target_angle = target_angles[joint]
        output_angle = np.around(output_angles[joint], 3)
        assert (target_angle == output_angle).all(), \
            f'Expected {joint} angle to be {target_angle} but got {output_angle}'

def test_get_base_skeleton():
    target_base_skeleton = {
        'hips': np.array([0, 0, 0]), 
        'lefthip': np.array([0.17614683, 0., 0.]), 
        'righthip': np.array([-0.17614683,  0.,  0.]), 
        'leftknee': np.array([ 0., -0.76974663,  0.]), 
        'rightknee': np.array([ 0., -0.76974663,  0.]), 
        'leftfoot': np.array([ 0., -0.69001824,  0.]), 
        'rightfoot': np.array([ 0., -0.69001824,  0.]), 
        'leftshoulder': np.array([0.32118727, 0., 0.]), 
        'rightshoulder': np.array([-0.32118727,  0.,  0.]), 
        'leftelbow': np.array([0.55494827, 0., 0.]), 
        'rightelbow': np.array([-0.55494827,  0.,  0.]), 
        'leftwrist': np.array([0.55303607, 0., 0.]), 
        'rightwrist': np.array([-0.55303607,  0.,  0.]), 
        'neck': np.array([0., 1., 0.])}

    kp_obj = KeypointRotations()
    new_kpts = kp_obj.add_neck_and_hip_keypoints(kpts_sample)
    new_kpts = kp_obj.center_keypoints(new_kpts, 'hips')

    bone_lengths = kp_obj.get_bone_lengths(new_kpts)
    normalization = bone_lengths['neck']
    base_skeleton = kp_obj.get_base_skeleton(bone_lengths, normalization)

    for joint in target_base_skeleton:
        target_length = np.around(target_base_skeleton[joint], 3)
        output_length = np.around(base_skeleton[joint], 3)
        assert (target_length == output_length).all(), \
            f'Expected {joint} length to be {target_length} but got {output_length}'

def test_reconstruct_joint_kpts_from_angles():
    target_leftknee_kpts = {
        'kpt1': np.array([0.11195379, 1.51792921, 1.70323553]), 
        'kpt2': np.array([-0.65639566,  1.53353214,  1.65958308])
    }

    kp_obj = KeypointRotations()
    new_kpts = kp_obj.add_neck_and_hip_keypoints(kpts_sample)

    kpts_rotations = {
        'hips': np.array([-1.55299538, -0.09598362,  0.79955135]),
        'lefthip': np.array([ 0.08336282,  0.13366153, -0.00558273]),
        'leftknee': np.array([-0.0337819 ,  0.83998899,  0.01508699]),
    }

    joint = 'leftknee'

    bone_lengths = kp_obj.get_bone_lengths(new_kpts)
    normalization = bone_lengths['neck']
    base_skeleton = kp_obj.get_base_skeleton(bone_lengths, normalization)

    output_kpt1, output_kpt2 = kp_obj.reconstruct_joint_kpts_from_angles(
        joint, kpts_rotations, base_skeleton, new_kpts['hips'], normalization
    )

    for target_kpt, output_kpt in zip(target_leftknee_kpts.values(), [output_kpt1, output_kpt2]):
        target_kpt = np.around(target_kpt, 3)
        output_kpt = np.around(output_kpt, 3)
        assert (target_kpt == output_kpt).all(), \
            f'Expected keypoint to be {target_kpt} but got {output_kpt}'