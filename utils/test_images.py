import numpy as np
import pandas as pd

import time
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

from pathlib import Path

from PIL import Image
import cv2

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))




if __name__ == "__main__":

    PATH_TO_MODEL_DIR = '/Users/ajeffries/Documents/School/ROB_535/Tensorflow/workspace/gta_project/exported-models/my_one_class_classifier_120421'
    PATH_TO_LABELS = '/Users/ajeffries/Documents/School/ROB_535/Tensorflow/workspace/gta_project/annotations/label_map.pbtxt'

    PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"
    test_data_location = '/Users/ajeffries/Documents/School/ROB_535/Tensorflow/workspace/gta_project/test'
    IMAGE_PATHS = [str(image_file) for image_file in Path(test_data_location).glob('**/*.jpg')]

    GUIPaths = [image_file.parent.stem + "/" + image_file.stem[:-10] for image_file in Path(test_data_location).glob('**/*.jpg')]

    print('Loading model...', end='')
    start_time = time.time()

    # Load saved model and build the detection function
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))

    XTest = []

    YTest =np.array(GUIPaths)
    np.save('YTestRegions.npy',YTest)

    # for image_path in IMAGE_PATHS:

    #     print('Running inference for {}... '.format(image_path))

    #     image_np = load_image_into_numpy_array(image_path)
    #     h,w, c = image_np.shape
        
    #     # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    #     input_tensor = tf.convert_to_tensor(image_np)

    #     input_tensor = input_tensor[tf.newaxis, ...]
    #     detections = detect_fn(input_tensor)

    #     # Outputs Detections
        
    #     num_detections = int(detections.pop('num_detections'))
    #     detections = {key: value[0, :num_detections].numpy()
    #                for key, value in detections.items()}
    #     detections['num_detections'] = num_detections

    #     # detection_classes should be ints.
    #     detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    #     # Grab The Box with the Highest Score
    #     idx = detections['detection_scores'].argmax()

        
    #     box = detections['detection_boxes'][idx]
    #     min_y, min_x,max_y, max_x = int(box[0]*h) , int(box[1]*w), int(box[2]*h),int(box[3]*w)

    #     # Crop Image
    #     img_cropped = image_np[min_y:max_y,min_x:max_x,:]
    #     # Resize Image
    #     img_cropped = cv2.resize(img_cropped, (100,100), interpolation = cv2.INTER_AREA)

    #     # Add Image to Test Array
    #     XTest.append(img_cropped)
    
    # XTest = np.array(XTest)

    # np.save('XTestRegions.npy',XTest)

    