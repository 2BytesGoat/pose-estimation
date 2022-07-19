import numpy as np

import keypoints.utils as utils

HIERARCHY = {'hips': [],
    'lefthip': ['hips'], 'leftknee': ['lefthip', 'hips'], 'leftfoot': ['leftknee', 'lefthip', 'hips'],
    'righthip': ['hips'], 'rightknee': ['righthip', 'hips'], 'rightfoot': ['rightknee', 'righthip', 'hips'],
    'neck': ['hips'],
    'leftshoulder': ['neck', 'hips'], 'leftelbow': ['leftshoulder', 'neck', 'hips'], 'leftwrist': ['leftelbow', 'leftshoulder', 'neck', 'hips'],
    'rightshoulder': ['neck', 'hips'], 'rightelbow': ['rightshoulder', 'neck', 'hips'], 'rightwrist': ['rightelbow', 'rightshoulder', 'neck', 'hips']
}

OFFSETS = {
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

class KeypointRotations:
    def __init__(self):
        self.kpts_hierarchy = HIERARCHY
        self.kpts_offsets = OFFSETS
        self.max_connected_joints = max([len(connected_joints) for connected_joints in self.kpts_hierarchy.values()])

    @staticmethod
    def rotate_pose(kpts:dict, axis:str, degrees:int=90)->dict:
        kpts_cpy = kpts.copy()
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

        kpts_cpy = kpts.copy()
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
        kpts_cpy = kpts.copy()
        root_pnt = kpts_cpy[root_pnt_name]
        for joint in kpts_cpy.keys():
            kpts_cpy[joint] -= root_pnt
        return kpts_cpy

    def _init_bone_lengths(self, kpts):
        bone_lengths = {}
        for joint in kpts:
            if joint == 'hips': continue
            parent = self.kpts_hierarchy[joint][0]

            _bone = kpts[joint] - kpts[parent]
            _bone_lengths = np.sqrt(np.sum(np.square(_bone), axis = -1)) # maybe replace with magnitude

            _bone_length = np.median(_bone_lengths)
            bone_lengths[joint] = _bone_length
        return bone_lengths

    def _init_kpts_rotations(self, kpts):
        kpts_rotations = {joint: np.array([0.,0.,0.]) for joint in kpts.keys()}
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

    def calculate_keypoint_angles(self, kpts):
        kpts = self.add_neck_and_hip_keypoints(kpts)
        kpts = self.center_keypoints(kpts, 'hips')

        kpts_rotations = self._init_kpts_rotations(kpts)
        kpts_rotations['hips'] = self._calculate_root_rotation(
            kpts['hips'], kpts['lefthip'], kpts['neck']
        )
        
        # cannot calculate angle if we don't have at least two connections
        for depth in range(2, self.max_connected_joints):    
            for joint, connected_joints in self.kpts_hierarchy.items():
                if len(connected_joints) != depth: 
                    continue
                parent = connected_joints[0]
                joint_rotation = self.get_joint_rotations(kpts[joint], self.kpts_offsets[joint], kpts[parent], connected_joints, kpts_rotations)
                kpts_rotations[parent] = joint_rotation

        return kpts_rotations

if __name__ == '__main__':
    def read_keypoints(filename):
        num_keypoints = 12
        fin = open(filename, 'r')

        kpts = []
        while(True):
            line = fin.readline()
            if line == '': break

            line = line.split()
            line = [float(s) for s in line]

            line = np.reshape(line, (num_keypoints, -1))
            kpts.append(line)

        kpts = np.array(kpts)
        return kpts

    def convert_to_dictionary(kpts):
        #its easier to manipulate keypoints by joint name
        keypoints_to_index = {'lefthip': 6, 'leftknee': 8, 'leftfoot': 10,
                            'righthip': 7, 'rightknee': 9, 'rightfoot': 11,
                            'leftshoulder': 0, 'leftelbow': 2, 'leftwrist': 4,
                            'rightshoulder': 1, 'rightelbow': 3, 'rightwrist': 5}

        kpts_dict = {}
        for key, k_index in keypoints_to_index.items():
            kpts_dict[key] = kpts[:,k_index]

        kpts_dict['joints'] = list(keypoints_to_index.keys())

        return kpts_dict

    kpts = read_keypoints('tests/kpts_3d.dat')
    kpts = convert_to_dictionary(kpts)

    kpts_sample = {
        key: kpts[key][0] for key in kpts['joints']
    }

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
        'hips': np.array([ 0.71293411,  9.59914047, 10.69924206]), 
        'neck': np.array([ 6.53495033,  9.70278881, 10.13816696])
    }

    calculator = KeypointRotations()
    angles = calculator.calculate_keypoint_angles(kpts_sample)

    print(angles['hips'])

    with open('tests/test_output_angles.txt', 'r') as f:
        test_angles = eval(f.read())
    
    print(test_angles['hips'])
    