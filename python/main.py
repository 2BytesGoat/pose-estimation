# %%
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

import socket
import numpy as np

class GodotUDPClient:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)

    def send_message(self, angles):
        message_str = str(angles).replace("'", '"')
        self.sock.sendto(message_str.encode('utf-8'), ("127.0.0.1", 4240))

def put_text_on_image(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, (10,10), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    return image

def calculate_plane(points):
    A = A = np.hstack((points[:,:2], np.ones((len(points),1))))
    b = points[:,2]
    return np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), b)

def get_angles(plane):
    oZ = np.subtract([0, 0, 0],[0, 0, 1])
    oY = np.subtract([0, 0, 0],[0, 1, 0])
    oX = np.subtract([0, 0, 0],[1, 0, 0])
    def get_angle(plane, vector):
        denom = np.sum(plane ** 2) ** 0.5 * np.sum(vector ** 2) ** 0.5
        angle = np.arcsin(abs(np.sum(plane * vector)) / denom)
        return angle
    return get_angle(plane, oX), get_angle(plane, oY), get_angle(plane, oZ)

angles = {
    "face": {
        "pitch": 0,
        "yaw": 0,
        "roll": 0
    }
}

# For webcam input:
client = GodotUDPClient()
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    if results.pose_landmarks is not None:
        points = []
        points_3d = []
        im_h, im_w, _ = image.shape
        for lm in results.pose_landmarks.landmark:
            points.append([lm.x, lm.y, lm.z])
        for i in [0,2,5,9,10]:
            lm, lm_w = results.pose_landmarks.landmark[i], results.pose_world_landmarks.landmark[i]
            # points_2d.append([lm.x * im_w, lm.y * im_h])
            points_3d.append([lm.x, lm.y, lm.z])
        
        points_3d = np.array(points_3d)
        face_plane = calculate_plane(points_3d)
        yaw, pitch, roll = get_angles(face_plane)

        angles["face"] = {
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll
        }
        client.send_message(angles)

    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
