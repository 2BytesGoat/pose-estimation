import cv2

def resize_and_pad(frame, shape=(192, 192)):
    height, width, _ = frame.shape
    top, bot, left, right = 0, 0, 0, 0
    if height > width:
        left = right = (height - width) // 2
    else:
        top = bot = (width - height) // 2
    padded = cv2.copyMakeBorder(frame, top, bot, left, right, cv2.BORDER_CONSTANT)

    return cv2.resize(padded, shape, interpolation = cv2.INTER_AREA)