import cv2

KEYPOINT_MAPPING = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5, 
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

def draw_keypoints(image, keypoints):
    points = []
    frameHeight, frameWidth, _ = image.shape
    for (y, x, conf) in keypoints:
        # Slice heatmap of corresponging body's part.
        x, y = frameWidth * x, frameHeight * y
        points.append((int(x), int(y)) if conf > 0.2 else None)
    for point in points:
        if point:
            cv2.ellipse(image, point, (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
    return image
