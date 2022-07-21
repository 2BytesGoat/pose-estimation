# %% 
import numpy as np

data_path = 'data/imgs/img0.txt'
with open(data_path, 'r') as f:
    data = np.array(eval(f.read()))
# %%
import mediapipe as mp
mp_keypoints = mp.solutions.pose.PoseLandmark

from keypoints import KeypointRotations

offset_directions = {
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

# %%
import socket

class GodotUDPClient:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)

    def send_message(self, angles):
        message_str = str(angles).replace("'", '"')
        self.sock.sendto(message_str.encode('utf-8'), ("127.0.0.1", 4240))

client = GodotUDPClient()
# %%
godot_mapping = {
    'LeftUpLeg': 'lefthip',
    'LeftLeg': 'leftknee',
    'LeftFoot': 'leftfoot',

    'RightUpLeg': 'righthip',
    'RightLeg': 'rightknee',
    'RightFoot': 'rightfoot',

    'LeftShoulder': 'leftshoulder',
    'LeftArm': 'leftelbow',
    'LeftForeArm': 'leftwrist',

    'RightShoulder': 'rightshoulder',
    'RightArm': 'rightelbow',
    'RightForeArm':'rightwrist',
}

kpts = {
    key: data[value] for key, value in offset_directions.items()
}

calculator = KeypointRotations()
angles = calculator.calculate_keypoint_angles(kpts)

message = {}

for key, value in godot_mapping.items():
    pitch, yaw, roll = angles[value]
    message[key] = {
        'pitch': pitch,
        'yaw': yaw,
        'roll': roll,
    }

client.send_message(message)
# %%
