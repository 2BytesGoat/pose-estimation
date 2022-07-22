'''Code adapted from Temuge Batpurev
link: https://github.com/TemugeB/joint_angles_calculate/blob/main/calculate_joint_angles.py
'''
import copy
import numpy as np

import keypoints.utils as utils

HIERARCHY = {
    'hips': [],
    'lefthip': ['hips'], 'leftknee': ['lefthip', 'hips'], 'leftfoot': ['leftknee', 'lefthip', 'hips'],
    'righthip': ['hips'], 'rightknee': ['righthip', 'hips'], 'rightfoot': ['rightknee', 'righthip', 'hips'],
    'neck': ['hips'],
    'leftshoulder': ['neck', 'hips'], 'leftelbow': ['leftshoulder', 'neck', 'hips'], 'leftwrist': ['leftelbow', 'leftshoulder', 'neck', 'hips'],
    'rightshoulder': ['neck', 'hips'], 'rightelbow': ['rightshoulder', 'neck', 'hips'], 'rightwrist': ['rightelbow', 'rightshoulder', 'neck', 'hips']
}

OFFSETS = {
    'hips': np.array([0, 0, 0]),
    'lefthip': np.array([1, 0, 0]), 'leftknee': np.array([0, -1, 0]), 'leftfoot': np.array([0, -1, 0]),
    'righthip': np.array([-1, 0, 0]), 'rightknee': np.array([0, -1, 0]), 'rightfoot': np.array([0, -1, 0]), 
    'neck': np.array([0, 1, 0]),
    'leftshoulder': np.array([1, 0, 0]), 'leftelbow': np.array([1, 0, 0]), 'leftwrist': np.array([1, 0, 0]),
    'rightshoulder': np.array([-1,0,0]), 'rightelbow': np.array([-1,0,0]), 'rightwrist': np.array([-1,0,0])
}

class KeypointRotations:
    def __init__(self):
        self.kpts_hierarchy = HIERARCHY
        self.kpts_offsets = OFFSETS
        self.max_connected_joints = max([len(connected_joints) for connected_joints in self.kpts_hierarchy.values()])

    @staticmethod
    def rotate_pose(kpts:dict, axis:str, degrees:int=90)->dict:
        kpts_cpy = copy.deepcopy(kpts)
        angle = np.deg2rad(degrees)
        if axis == 'x':
            R = utils.get_R_x(angle)
        elif axis == 'y':
            R = utils.get_R_y(angle)
        elif axis == 'z':
            R = utils.get_R_z(angle)
        else:
            raise NotImplementedError

        for joint in kpts_cpy:
            kpts_cpy[joint] = R @ kpts_cpy[joint]

        return kpts_cpy

    @staticmethod
    def add_neck_and_hip_keypoints(kpts:dict)->dict:
        def calculate_midpoint(keypoint_left:np.array, keypoint_right:np.array)->np.array:
            half_distance = (keypoint_left - keypoint_right) / 2
            keypoint_mid = keypoint_right + half_distance
            return keypoint_mid

        kpts_cpy = copy.deepcopy(kpts)
        kpts_cpy['neck'] = calculate_midpoint(
            kpts['leftshoulder'], 
            kpts['rightshoulder']
        )
        kpts_cpy['hips'] = calculate_midpoint(
            kpts['lefthip'], 
            kpts['righthip']
        )
        return kpts_cpy

    @staticmethod
    def _calculate_root_rotation(root_pnt, root_pnt_x, root_pnt_y):
        # calculate unit vectors of root joint
        root_u = utils.calculate_unit_vector(root_pnt_x, root_pnt)
        root_v = utils.calculate_unit_vector(root_pnt_y, root_pnt)
        root_w = np.cross(root_u, root_v)

        # calculate the rotation matrix
        C = np.stack([root_u, root_v, root_w], axis=1)
        theta_z, theta_y, theta_x = utils.Decompose_R_ZXY(C)
        root_rotation = np.array([theta_z, theta_x, theta_y])
        return root_rotation

    @staticmethod
    def center_keypoints(kpts, root_pnt_name):
        kpts_cpy = copy.deepcopy(kpts)
        root_pnt = kpts_cpy[root_pnt_name]
        for joint in kpts_cpy.keys():
            kpts_cpy[joint] -= root_pnt
        return kpts_cpy

    def _init_kpts_rotations(self, kpts):
        kpts_rotations = {joint: np.array([0.,0.,0.]) for joint in kpts.keys()}
        return kpts_rotations

    def reconstruct_joint_kpts_from_angles(self, joint, angles, base_skeleton, root_pnt, normalization):
        # joint, kpts_rotations, base_skeleton, new_kpts['hips'], normalization
        
        joint_hierarchy = self.kpts_hierarchy[joint]
        #get the current position of the parent joint
        r1 = root_pnt / normalization
        for parent in joint_hierarchy:
            if parent == 'hips': continue
            R = self.get_rotation_chain(self.kpts_hierarchy[parent], angles)
            r1 = r1 + R @ base_skeleton[parent]
        #get the current position of the joint. Note: r2 is the final position of the joint. r1 is simply calculated for plotting.
        r2 = r1 + self.get_rotation_chain(joint_hierarchy, angles) @ base_skeleton[joint]

        return r1, r2

    def get_bone_lengths(self, kpts):
        bone_lengths = {}
        for joint in kpts:
            if joint == 'hips': continue
            parent = self.kpts_hierarchy[joint][0]

            _bone = kpts[joint] - kpts[parent]
            _bone_lengths = np.sqrt(np.sum(np.square(_bone), axis = -1)) # maybe replace with magnitude

            _bone_length = np.median(_bone_lengths)
            bone_lengths[joint] = _bone_length
        return bone_lengths

    def get_base_skeleton(self, body_lengths, normalization):
        def _set_length(joint_type):
            base_skeleton['left' + joint_type] = self.kpts_offsets['left' + joint_type] * ((body_lengths['left' + joint_type] + body_lengths['right' + joint_type])/(2 * normalization))
            base_skeleton['right' + joint_type] = self.kpts_offsets['right' + joint_type] * ((body_lengths['left' + joint_type] + body_lengths['right' + joint_type])/(2 * normalization))
        base_skeleton = {'hips': np.array([0,0,0])}
        base_skeleton['neck'] = self.kpts_offsets['neck'] * (body_lengths['neck']/normalization)
        _set_length('hip')
        _set_length('knee')
        _set_length('foot')
        _set_length('shoulder')
        _set_length('elbow')
        _set_length('wrist')
        return base_skeleton

    #helper function that composes a chain of rotation matrices
    def get_rotation_chain(self, hierarchy, frame_rotations, inverse=False):
        if not inverse:
            hierarchy = hierarchy[::-1]
        #this code assumes ZXY rotation order
        R = np.eye(3)
        for parent in hierarchy:
            _r_angles = frame_rotations[parent]
            _R = utils.get_R_z(_r_angles[0])@utils.get_R_x(_r_angles[1])@utils.get_R_y(_r_angles[2])
            R = R @ _R.T if inverse else R @ _R
        return R

    def get_joint_rotations(self, joint_pos, joint_offset, parent_pos, connected_joints, kpts_rotations):
        _invR = self.get_rotation_chain(connected_joints[1:], kpts_rotations, True)
        b = _invR @ (joint_pos - parent_pos)

        _R = utils.Get_R2(joint_offset, b)
        tz, ty, tx = utils.Decompose_R_ZXY(_R)
        joint_rs = np.array([tz, ty, tx])

        return joint_rs

    def calculate_keypoint_angles(self, kpts):
        kpts = self.add_neck_and_hip_keypoints(kpts)
        kpts = self.center_keypoints(kpts, 'hips')

        kpts_rotations = self._init_kpts_rotations(kpts)
        kpts_rotations['hips'] = self._calculate_root_rotation(
            kpts['hips'], kpts['lefthip'], kpts['neck']
        )
        
        # cannot calculate angle if we don't have at least two connections
        for depth in range(2, self.max_connected_joints+1):    
            for joint, connected_joints in self.kpts_hierarchy.items():
                if len(connected_joints) != depth: 
                    continue
                parent = connected_joints[0]
                z, y, x = self.get_joint_rotations(kpts[joint], self.kpts_offsets[joint], kpts[parent], connected_joints, kpts_rotations)
                kpts_rotations[parent] = np.array([z, x, y])

        return kpts_rotations
