import cv2
import mediapipe as mp

from pathlib import Path

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For static images:
IMAGE_FILES = ['data/imgs/img0.jpg', 'data/imgs/img1.jpg', 'data/imgs/img2.jpg']
BG_COLOR = (192, 192, 192) # gray
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5) as pose:
    
    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        image_height, image_width, _ = image.shape
        
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            continue

        points = []
        for lm in results.pose_world_landmarks.landmark:
            points.append([lm.x, lm.y, lm.z])
        
        txt_name = Path(file).parent / (Path(file).stem + '.txt')
        with open(txt_name, 'w') as f:
            f.write(str(points))

        annotated_image = image.copy()
        png_name = str(Path(file).parent / (Path(file).stem + '_anno.png'))
        # Draw pose landmarks on the image.
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.imwrite(png_name, annotated_image)
