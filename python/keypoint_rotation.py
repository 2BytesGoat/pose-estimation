from typing import List

import numpy as np

import utils


class KeypointRotations:
    def __init__(self):
        self.hierarchy = {
            'hips': [],
            'lefthip': ['hips'], 'leftknee': ['lefthip', 'hips'], 'leftfoot': ['leftknee', 'lefthip', 'hips'],
            'righthip': ['hips'], 'rightknee': ['righthip', 'hips'], 'rightfoot': ['rightknee', 'righthip', 'hips'],
            'neck': ['hips'],
            'leftshoulder': ['neck', 'hips'], 'leftelbow': ['leftshoulder', 'neck', 'hips'], 'leftwrist': ['leftelbow', 'leftshoulder', 'neck', 'hips'],
            'rightshoulder': ['neck', 'hips'], 'rightelbow': ['rightshoulder', 'neck', 'hips'], 'rightwrist': ['rightelbow', 'rightshoulder', 'neck', 'hips']
        }
        self.max_connected_joints = max([len(connected_joints) for connected_joints in self.hierarchy.values()])
    
    def _calculate_midpoint(self, keypoint_left: List, keypoint_right: List)->List:
        half_distance = np.subtract(keypoint_left, keypoint_right) / 2
        keypoint_mid = np.sum([keypoint_right, half_distance], axis=0)
        return keypoint_mid

    def _calculate_root_rotation(self, root_pnt, root_pnt_x, root_pnt_y):
        # calculate unit vectors of root joint
        root_u = utils.calculate_unit_vector(root_pnt_x, root_pnt)
        root_v = utils.calculate_unit_vector(root_pnt_y, root_pnt)
        root_w = np.cross(root_u, root_v)

        # calculate the rotation matrix
        C = np.stack([root_u, root_v, root_w], axis=1)
        theta_z, theta_y, theta_x = utils.Decompose_R_ZXY(C)
        root_rotation = np.array([theta_z, theta_x, theta_y])
        return root_rotation

    def _center_keypoints(self, kpts, root_pnt_name='hip'):
        kpts_cpy = kpts.copy()
        root_pnt = kpts_cpy[root_pnt_name]
        for joint in kpts_cpy.keys():
            kpts_cpy[joint] -= root_pnt
        return kpts_cpy

    def add_neck_and_hip_keypoints(self, kpts):
        kpts_cpy = kpts.copy()
        kpts_cpy['neck'] = self._calculate_midpoint(
            kpts['leftshoulder'], 
            kpts['rightshoulder']
        )
        kpts_cpy['hips'] = self._calculate_midpoint(
            kpts['lefthip'], 
            kpts['righthip']
        )
        return kpts_cpy

    def _init_bone_lengths(self, kpts):
        bone_lengths = {}
        for joint in kpts:
            if joint == 'hips': continue
            parent = self.hierarchy[joint][0]

            _bone = kpts[joint] - kpts[parent]
            _bone_lengths = np.sqrt(np.sum(np.square(_bone), axis = -1)) # maybe replace with magnitude

            _bone_length = np.median(_bone_lengths)
            bone_lengths[joint] = _bone_length
        return bone_lengths

    def _init_offset_directions(self):
        offset_directions = {
            'hips': np.array([0, 0, 0]),

            'lefthip': np.array([1, 0, 0]),
            'leftknee': np.array([0, -1, 0]),
            'leftfoot': np.array([0, -1, 0]),

            'righthip': np.array([-1, 0, 0]),
            'rightknee': np.array([0, -1, 0]),
            'rightfoot': np.array([0, -1, 0]), 

            'neck': np.array([0, 1, 0]),

            'leftshoulder': np.array([1, 0, 0]),
            'leftelbow': np.array([1, 0, 0]),
            'leftwrist': np.array([1, 0, 0]),

            'rightshoulder': np.array([-1,0,0]),
            'rightelbow': np.array([-1,0,0]),
            'rightwrist': np.array([-1,0,0])
        }
        return offset_directions

    def _init_kpts_rotations(self, kpts):
        kpts_rotations = {joint: [0.,0.,0.] for joint in kpts.keys()}
        return kpts_rotations

    def get_joint_rotations(self, joint_pos, joint_offset, parent_pos, connected_joints, kpts_rotations):
        _invR = np.eye(3)
        
        # first joint corresponds to parent joint, the rest are grandparents
        for grampa_name in connected_joints[1:]:
            _r_angles = kpts_rotations[grampa_name]
            R = utils.get_R_z(_r_angles[0]) @ utils.get_R_x(_r_angles[1]) @ utils.get_R_y(_r_angles[2])
            _invR = _invR @ R.T

        b = _invR @ (joint_pos - parent_pos)

        _R = utils.Get_R2(joint_offset, b)
        tz, ty, tx = utils.Decompose_R_ZXY(_R)
        joint_rs = np.array([tz, tx, ty])

        return joint_rs

    def calculate_joint_angles(self, kpts):
        kpts = self.add_neck_and_hip_keypoints(kpts)
        kpts = self._center_keypoints(kpts, 'hips')

        kpts_rotations = self._init_kpts_rotations(kpts)
        kpts_rotations['hips'] = self._calculate_root_rotation(
            kpts['hips'], kpts['lefthip'], kpts['neck']
        )

        kpts_offsets = self._init_offset_directions()
        
        # cannot calculate angle if we don't have at least two connections
        for depth in range(2, self.max_connected_joints):    
            for joint, connected_joints in self.hierarchy.items():
                if len(connected_joints) != depth: 
                    continue
                parent = connected_joints[0]
                joint_rotation = self.get_joint_rotations(kpts[joint], kpts_offsets[joint], kpts[parent], connected_joints, kpts_rotations)
                kpts_rotations[parent] = joint_rotation

        return kpts_rotations