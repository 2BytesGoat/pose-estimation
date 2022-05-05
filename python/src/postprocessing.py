import math

from .utils import KEYPOINT_MAPPING

def get_face_pitch(keypoints, bbox):
    y_center, _, _ = keypoints[KEYPOINT_MAPPING['nose']]
    _, y_top, _, y_bot = bbox
    # Get nose position relative to bbox height
    position = ((y_center - y_top) / (y_bot - y_top) - 0.5) * 2
    # Transform nose position into an angle
    angle = max(-90, min(position * 90, 90))
    return round(angle, 3)

def get_face_yaw(keypoints, bbox):
    _, x_center, _ = keypoints[KEYPOINT_MAPPING['nose']]
    x_left, _, x_right, _ = bbox
    # Get nose position relative to the ears
    position = ((x_center - x_left) / (x_right - x_left) - 0.5) * 2
    # Transform nose position into an angle
    angle = max(-90, min(position * 90, 90))
    return round(angle, 3)

def to_rads(angle):
    return angle / 180 * math.pi

def calculate_face_angles(keypoints, results, in_rads=True):
    pitch, yaw, roll = 0, 0, 0
    if len(results.boxes) > 0:
        for box_idx in range(len(results.boxes)):
            bbox = results.boxes[box_idx]
        pitch = get_face_pitch(keypoints, bbox)
        yaw = get_face_yaw(keypoints, bbox)
    if in_rads:
        return list(map(to_rads, [pitch, yaw, roll]))
    return [pitch, yaw, roll]

def get_angle(keypoints, limb1, limb2, invert=False, shift=0):
    y_limb1, x_limb1, _ = keypoints[KEYPOINT_MAPPING[limb1]]
    y_limb2, x_limb2, _ = keypoints[KEYPOINT_MAPPING[limb2]]
    angle_tan = (y_limb2 - y_limb1) / (x_limb2 - x_limb1)
    rads = max(-90, min(math.atan(angle_tan), 90))
    return (rads * -1 if invert else rads) + shift

def calculate_arm_angles(keypoints):
    right_arm = get_angle(keypoints, 'right_shoulder', 'right_elbow', shift=-1.5, invert=True)
    left_arm = get_angle(keypoints, 'left_shoulder', 'left_elbow', shift=-1.5)
    right_hand = get_angle(keypoints, 'right_elbow', 'right_wrist')
    left_hand = get_angle(keypoints, 'left_elbow', 'left_wrist')
    return [right_arm, left_arm, right_hand, left_hand]