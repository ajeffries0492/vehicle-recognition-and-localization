import os
from pathlib import Path
import pathlib
import tensorflow as tf
from tensorflow import keras
import matplotlib.patches as patches
import cv2
import time

from PIL import Image

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from object_detection.utils import label_map_util

def find_detections_and_centroid_in_image(imageFileLocation,detect_fn,category_index=None):

    image = np.array(Image.open(imageFileLocation))
    H,W, _ = image.shape


    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    max_score = detections['detection_scores'][0]
    ymin,xmin,ymax,xmax = detections['detection_boxes'][0]
    label = detections['detection_classes'][0]

    ymin,xmin,ymax,xmax = int(ymin*H),int(xmin*W),int(ymax*H),int(xmax*W)
    
    if category_index is not None:
        label = category_index[label]['name']

    text = f"Class {label} {100*max_score:.2f}%"
    print(text)

    image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (50, 168, 164), 4)
    
    cent_x, cent_y = int((xmax + xmin) /2), int((ymax + ymin) /2)
    
    cv2.putText(image, text, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 168, 164), 2) 
    
    return image, cent_x, cent_y
def find_detections_in_image(imageFileLocation,detect_fn,class_model=None,
                             dim_model=(100,100),category_index=None,saveFile=False):

    image = np.array(Image.open(imageFileLocation))
    H,W, _ = image.shape


    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    max_score = detections['detection_scores'][0]
    ymin,xmin,ymax,xmax = detections['detection_boxes'][0]
    label = detections['detection_classes'][0]

    ymin,xmin,ymax,xmax = int(ymin*H),int(xmin*W),int(ymax*H),int(xmax*W)
    
    if class_model is not None:
        # Crop Full Sized Ground Truth Image
        img_cropped = image[ymin:ymax,xmin:xmax,:]
        # Resize
        img_cropped = cv2.resize(img_cropped, dim_model, interpolation = cv2.INTER_AREA)
        img_cropped = np.expand_dims(img_cropped,axis=0)
        # Run Classification
        y_pred = class_model(img_cropped)
        label = np.argmax(y_pred) + 1
        max_score = np.max(y_pred)
    
    if category_index is not None:
        label = category_index[label]['name']

    text = f"Class {label} {100*max_score:.2f}%"


    image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (50, 168, 164), 4)
    cv2.putText(image, text, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 168, 164), 2) 
    plt.figure(figsize=(15,10))
    plt.imshow(image)
    plt.title(text)
    
    
    filename = Path(imageFileLocation).stem
    filename = f"{filename}Evaluated.png"
    if saveFile:
        plt.tight_layout()
        plt.savefig(filename)
    plt.show()
    


def get_keras_model(PATH_TO_SAVED_MODEL):
    print('Loading keras model ...', end='')
    start_time = time.time()

    # Load saved model
    model = keras.models.load_model(PATH_TO_SAVED_MODEL)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))
    return model
    
def find_detections_in_array_of_images(imageFileLocations,detect_fn,class_model=None,category_index=None,grid_x=2):
    
    N = len(imageFileLocations)
    
    grid_y = np.ceil(N/grid_x).astype(int)
    
    figure, ax = plt.subplots(grid_x,grid_y,figsize=(25,15))
    
    plot_grid = np.meshgrid(np.arange(grid_x), np.arange(grid_y))
    
    plot_grid_x = plot_grid[0].flatten()
    plot_grid_y = plot_grid[1].flatten()
    for idx,imageFileLocation in enumerate(imageFileLocations):

        image = np.array(Image.open(imageFileLocation))
        H,W, _ = image.shape


        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        max_score = detections['detection_scores'][0]
        ymin,xmin,ymax,xmax = detections['detection_boxes'][0]
        label = detections['detection_classes'][0]

        ymin,xmin,ymax,xmax = int(ymin*H),int(xmin*W),int(ymax*H),int(xmax*W)
        
        if class_model is not None:
            # Crop Full Sized Ground Truth Image
            img_cropped = image[ymin:ymax,xmin:xmax,:]
            # Resize
            img_cropped = cv2.resize(img_cropped, dim_model, interpolation = cv2.INTER_AREA)
            img_cropped = np.expand_dims(img_cropped,axis=0)
            # Run Classification
            y_pred = class_fn(img_cropped)
            label = np.argmax(y_pred) + 1
            max_score = np.max(y_pred)

        if category_index is not None:
            label = category_index[label]['name']

        text = f"Class {label} {100*max_score:.2f}%"
        

        image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (50, 168, 164), 4)
        cv2.putText(image, text, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 168, 164), 2) 
        
        i_plot_x = plot_grid_x[idx]
        
        i_plot_y = plot_grid_y[idx]
        
        ax[i_plot_x,i_plot_y].imshow(image)
        ax[i_plot_x,i_plot_y].title.set_text(text)
        

    
    plt.show()
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


if __name__ == "__main__":
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
    output_location = args.output_location
    

    PATH_TO_MODEL_DIR = f'{pathToModelDir}/{MODEL_NAME}'
    PATH_TO_LABELS = f'{pathToModelDir}/{MODEL_NAME}/label_map.pbtxt'

    # Load Parquet with BBOX PRediction
    df_test_predictions = pd.read_parquet(PATH_TO_MODEL_DIR/Path(f'XTest_{MODEL_NAME}_BBOX.parquet'))
    




    task2_info = []
    pointType = ['x','y','z']
    for i,row in df_test_predictions.iterrows():


        # Add Image Location
        imageLoc = row['guid/image']

        # Specify Image and Regions File to Look at
        imageFileLocation = str(Path(testDataLocation) / Path(f"{imageLoc}_image.jpg"))

        if i == 0:
            # Open Imsge to Get Info on Shape
            img = np.array(Image.open(imageFileLocation))
            # Get Image Shape
            imageH,imageW, _ = img.shape



        xyz = np.fromfile(imageFileLocation.replace('_image.jpg', '_cloud.bin'), dtype=np.float32)
        xyz = xyz.reshape([3, -1])

        proj = np.fromfile(imageFileLocation.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
        proj.resize([3, 4])

        # Lidar Points in Pixels 
        uv = proj @ np.vstack([xyz, np.ones_like(xyz[0, :])])
        uv = uv / uv[2, :]


        # Get Projection Bounding Boxes
        xmin, ymin, xmax, ymax = int(row['xmin']*imageW),int(row['ymin']*imageH), int(row['xmax']*imageW), int(row['ymax']*imageH)
        # Find the Center of That Box
        cent_x, cent_y = int((xmax + xmin) /2), int((ymax + ymin) /2)
        centroid = np.array([cent_x, cent_y,1])

        # Find Closes Lidar Point By Checking Distance Between Projected Points and Center Points
        idx_centroid = np.argmin(np.linalg.norm(uv - centroid[:,None],axis=0))
        lidar_centroid = xyz[:,idx_centroid]

        for i_l,l in enumerate(lidar_centroid):
            imageCentrodinfo = {'guid/image/axis': imageLoc +"/" + pointType[i_l], 'value': l}
            task2_info.append(imageCentrodinfo)


    # Create DataFrame with Guesses
    df = pd.DataFrame(task2_info)
    

    # Save DataFrame
    # Save Answers
    pathToCSVResults = Path(output_location) / Path(f'{MODEL_NAME}_task2_localization.csv')
    df.loc[:,['guid/image/axis','value']].to_csv(pathToCSVResults,index=False)

