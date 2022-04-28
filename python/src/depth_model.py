from numpy import outer
import tensorflow as tf

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def format_image(image):
    img_resized = tf.image.resize(image, [256,256], method='bicubic', preserve_aspect_ratio=False)
    #img_resized = tf.transpose(img_resized, [2, 0, 1])
    img_input = img_resized.numpy()
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    img_input = (img_input - mean) / std
    reshape_img = img_input.reshape(1,256,256,3)
    tensor = tf.convert_to_tensor(reshape_img, dtype=tf.float32)
    return tensor

def scale_image_values(image):
    depth_min = image.min()
    depth_max = image.max()
    img_out = (255 * (image - depth_min) / (depth_max - depth_min)).astype("uint8")
    return img_out

class Midas:
    def __init__(self, model_path):
        self.model_path = model_path
        # Initialize the TFLite interpreter
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']

    def predict(self, image):
        tensor = format_image(image)
        # TF Lite format expects tensor type of float32.
        self.interpreter.set_tensor(self.input_details[0]['index'], tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        output = output.reshape(256, 256)
        return scale_image_values(output)