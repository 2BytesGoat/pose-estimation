import tensorflow as tf

def format_image(image):
    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    return tf.cast(tf.expand_dims(image, axis=0), dtype=tf.int32)

class Movenet:
    def __init__(self, model_path):
        self.model_path = model_path
        # Initialize the TFLite interpreter
        self.model = tf.lite.Interpreter(model_path=model_path)
        self.model.allocate_tensors()
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()

    def predict(self, image):
        _image = format_image(image)
        # TF Lite format expects tensor type of uint8.
        input_image = tf.cast(_image, dtype=tf.uint8)
        self.model.set_tensor(self.input_details[0]['index'], input_image)
        self.model.invoke()
        # Output is a [1, 1, 17, 3] numpy array.
        keypoints = self.model.get_tensor(self.output_details[0]['index'])
        return keypoints
