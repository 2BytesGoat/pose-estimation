import math

from .utils import KEYPOINT_MAPPING

def get_face_pitch(keypoints):
    # TODO: straight face should be 20 shifted
    y1_left, x1_left, _ = keypoints[KEYPOINT_MAPPING['left_ear']]
    y2_left, x2_left, _ = keypoints[KEYPOINT_MAPPING['left_eye']]
    y1_right, x1_right, _ = keypoints[KEYPOINT_MAPPING['right_ear']]
    y2_right, x2_right, _ = keypoints[KEYPOINT_MAPPING['right_eye']]

    m_left = (y2_left - y1_left) / (x2_left - x1_left)
    m_right = (y2_right - y1_right) / (x2_right - x1_right)
    rads = (abs(m_left) + abs(m_right)) / 2
    angle = max(-90, min(math.atan(rads) * 180 / math.pi, 90))
    sign = math.copysign(1, m_left)
    return round(angle * sign, 3)

def get_face_yaw(keypoints):
    _, x_left, _ = keypoints[KEYPOINT_MAPPING['left_ear']]
    _, x_right, _ = keypoints[KEYPOINT_MAPPING['right_ear']]
    _, x_center, _ = keypoints[KEYPOINT_MAPPING['nose']]

    # Get nose position relative to the ears
    position = ((x_center - x_right) / (x_left - x_right) - 0.5) * 2
    # Transform nose position into an angle
    angle = max(-90, min(position * 90, 90))
    return round(angle, 3)

def to_rads(angle):
    return angle / 180 * math.pi

def calculate_face_angles(keypoints, in_rads=True):
    pitch = get_face_pitch(keypoints)
    yaw = get_face_yaw(keypoints)
    if in_rads:
        return list(map(to_rads, [pitch, yaw, 0]))
    return pitch, yaw