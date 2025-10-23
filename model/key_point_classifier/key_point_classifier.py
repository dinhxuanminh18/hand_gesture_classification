import numpy as np
import tensorflow as tf

class KeyPointClassifier:
    def __init__(self, model_path = 'model/key_point_classifier/key_point_classifier.tflite'):
        self.interpreter = tf.lite.Interpreter(model_path = model_path, num_threads = 1)
        self.interpreter.allocate_tensors()

        self.get_input_detail = self.interpreter.get_input_details()
        self.get_output_detail = self.interpreter.get_output_details()

    def __cal__(self, landmark_list):
        self.interpreter.set_tensor(self.get_input_detail[0]['index'], np.array([landmark_list], dtype = np.float32))
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.get_output_detail[0]['index'])

        result = np.argmax(np.squeeze(output))

        return result
    
