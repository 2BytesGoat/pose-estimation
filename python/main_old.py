# %%
import cv2

from src.webcam import Webcam
from src.preprocessing import resize_and_pad
from src.postprocessing import calculate_face_angles
from src.pose_model import Movenet
from src.depth_model import Midas
from src.face_model import BlazeFaceDetector
from src.networking import GodotUDPClient
from src.utils import draw_keypoints

import time

def put_text_on_image(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, (10,10), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    return image

if __name__ == '__main__':
    cam = Webcam()
    cam.start_capture()

    model_keypoints = Movenet('models\movenet_float16\lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite')
    model_face = BlazeFaceDetector(type="back")

    client = GodotUDPClient()

    while True:
        frame = cam.grab_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        keypoint_frame = resize_and_pad(frame)
        keypoints = model_keypoints.predict(keypoint_frame)[0][0]
        frame = draw_keypoints(keypoint_frame, keypoints)

        results = model_face.detectFaces(keypoint_frame)
        model_face.drawDetections(frame, results)
        
        face_angles = calculate_face_angles(keypoints, results, in_rads=True)
        client.send_message(face_angles)

        text = f'{face_angles}'
        frame = cv2.resize(frame, (480, 480))

        frame = put_text_on_image(frame, text)        

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop_capture()
    cv2.destroyAllWindows()
# %%
