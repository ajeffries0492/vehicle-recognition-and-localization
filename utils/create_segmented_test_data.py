import os
from pathlib import Path
import pathlib
import tensorflow as tf

import cv2
import time

from PIL import Image

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from object_detection.utils import label_map_util

def get_test_image_detection_info(detections,guidfilename):
  num_detections = int(detections.pop('num_detections'))
  detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
  detections['num_detections'] = num_detections

  # detection_classes should be ints.
  detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

  max_score = detections['detection_scores'][0]
  ymin,xmin,ymax,xmax = detections['detection_boxes'][0]

  test_image_info = {'guid/image': guidfilename,'max_score': max_score,
    'label': detections['detection_classes'][0],
    #  'pred_class': category_index[detections['detection_classes'][0] ]['name'],
    'ymin': ymin,
    'xmin': xmin,
    'ymax': ymax,
    'xmax': xmax}

  return test_image_info
def get_detection_predictions(testDataLocation,detect_fn):
  test_image_predictions = []

  for idx,test_image_file_path in enumerate(Path(testDataLocation).glob('**/*.jpg')):

    guidfilename = test_image_file_path.parent.name + "/" + test_image_file_path.stem[:-6]


    image_np = np.array(Image.open(test_image_file_path))

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run Detection Algorithm
    detections = detect_fn(input_tensor)


    test_image_info = get_test_image_detection_info(detections,guidfilename)
    test_image_predictions.append(test_image_info)

  # Create Test Image Prediction Boxes 
  df_test_predictions = pd.DataFrame(test_image_predictions)
  return df_test_predictions

def get_model(PATH_TO_MODEL_DIR):
   # Load Model
  PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

  print('Loading model...', end='')
  start_time = time.time()

  # Load saved model and build the detection function
  detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

  end_time = time.time()
  elapsed_time = end_time - start_time
  print('Done! Took {} seconds'.format(elapsed_time))
  return detect_fn

if __name__ =="__main__":

  import argparse

  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('--testDataLocation', type=str,help='Path to image test data',default='../workspace/gta_project/test')
  parser.add_argument('--pathToModelDir', type=str,help='Path to model directory',default='../workspace/gta_project/exported-models')
  parser.add_argument('--modelName', type=str,help='Model to Use',default='three_class_resnet50_v1_fpn_120821')
  parser.add_argument('--imageDimSize', type=int,help='Dimensions of Image',default=120)

  args = parser.parse_args()
  testDataLocation = args.testDataLocation
  pathToModelDir = args.pathToModelDir
  MODEL_NAME = args.modelName
  imageDimSize = args.imageDimSize


  PATH_TO_MODEL_DIR = f'{pathToModelDir}/{MODEL_NAME}'
  PATH_TO_LABELS = f'{pathToModelDir}/{MODEL_NAME}/label_map.pbtxt'
  dim_train = (imageDimSize,imageDimSize)

  fileInfo = f'{imageDimSize}by{imageDimSize}SegmentedImages'

  # Get SSD Model
  detect_fn = get_model(PATH_TO_MODEL_DIR)
 

  # Get Predictions for all Test Images
  df_test_predictions = get_detection_predictions(testDataLocation,detect_fn)
  # Save CSV with Guess
  pathToCSVResults = Path(PATH_TO_MODEL_DIR) / Path(f"{PATH_TO_MODEL_DIR}.csv")
  df_test_predictions.loc[:,['guid/image','label']].to_csv(pathToCSVResults,index=False)
  # Save To Parquet for Later
  df_test_predictions.to_parquet(Path(PATH_TO_MODEL_DIR) / Path(f'XTest_{MODEL_NAME}_BBOX.parquet'))

  # Create Bounding Box Test Images
  imageData = []
  XGUID = []
  for i, row in df_test_predictions.iterrows():

      # Add Image Location
      imageLoc = row['filename']
      
      XGUID.append(imageLoc)
      
      
      # Specify Image and Regions File to Look at
      imageFileLocation = Path(testDataLocation) / Path(imageLoc)
      print(f"Processing Image File {i}: {imageLoc}")


      img = np.array(Image.open(imageFileLocation))
      # Get Image Shape
      imageH,imageW, _ = img.shape
      # Get Projection Bounding Boxes
      min_x, min_y, max_x, max_y = int(row['xmin']*imageW),int(row['ymin']*imageH), int(row['xmax']*imageW), int(row['ymax']*imageH)

      # Crop Full Sized Ground Truth Image
      img_cropped = img[min_y:max_y,min_x:max_x,:]

      # Resize
      img_cropped = cv2.resize(img_cropped, dim_train, interpolation = cv2.INTER_AREA)


      imageData.append(img_cropped) # Add Image


  XGUID = np.array(XGUID)
  np.save(f'XTestGUID{fileInfo}{MODEL_NAME}.npy',XGUID)

  imageData = np.array(imageData)

  print(imageData.shape)
  np.save(f'XTest{fileInfo}{MODEL_NAME}.npy',imageData)