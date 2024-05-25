import os
import threading

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter

tf.random.set_seed(1234)


class VideoStream:
    """Camera object that controls video streaming from the Picamera"""

    def __init__(self, resolution=(640, 480), framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True


GRAPH_NAME = 'detect.tflite'
MODEL_DIR = 'models'
LABELMAP_NAME = 'labelmap.txt'

min_conf_threshold = .2
resW, resH = "1280x720".split('x')
imW, imH = int(resW), int(resH)

# gpu_delegate = load_delegate('libtensorflowlite_gpu_delegate.dll')
#
# # Load the TFLite model with the GPU delegate
# interpreter = Interpreter(model_path=MODEL_DIR, experimental_delegates=[gpu_delegate])
# interpreter.allocate_tensors()

CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_DIR, GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_DIR, LABELMAP_NAME)

with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

if labels[0] == '???':
    del (labels[0])

interpreter = Interpreter(model_path=PATH_TO_CKPT)

# gpu_delegate = load_delegate('libtensorflowlite_gpu_delegate.dll')
# interpreter = Interpreter(model_path=MODEL_DIR, experimental_delegates=[gpu_delegate])

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
out_name = output_details[0]['name']

if ('StatefulPartitionedCall' in out_name):  # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:  # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2
