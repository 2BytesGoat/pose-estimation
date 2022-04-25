# %%
import cv2

from src.webcam import Webcam
from src.preprocessing import resize_and_pad
from src.postprocessing import calculate_face_angles
from src.pose_model import Movenet
from src.networking import GodotUDPClient
from src.utils import draw_keypoints

def put_text_on_image(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, (10,10), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    return image

if __name__ == '__main__':
    cam = Webcam()
    cam.start_capture()

    model = Movenet('models\movenet_float16\lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite')

    client = GodotUDPClient()

    while True:
        frame = cam.grab_frame()
        frame = resize_and_pad(frame)
        keypoints = model.predict(frame)[0][0]
    
        frame = draw_keypoints(frame, keypoints)
        angles = calculate_face_angles(keypoints, in_rads=True)
        client.send_message(angles)

        text = f'{angles}'
        frame = cv2.resize(frame, (480, 480))

        frame = put_text_on_image(frame, text)        

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop_capture()
    cv2.destroyAllWindows()
# %%
