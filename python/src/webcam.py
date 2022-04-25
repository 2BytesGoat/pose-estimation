import cv2

class Webcam:
    def __init__(self, camera_idx=0):
        self.vid = None
        self.camera_idx = camera_idx
    
    def start_capture(self):
        self.vid = cv2.VideoCapture(self.camera_idx)

    def stop_capture(self):
        if self.vid:
            self.vid.release()

    def grab_frame(self):
        ret, frame = self.vid.read()
        return frame

