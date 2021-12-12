import cv2
import pandas as pd
import numpy as np

from pathlib import Path
from PIL import Image


import time

def compute_intersection_over_union(boxA, boxB):
    
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def get_rectangular_box(vert_2D,imageW,imageH):


    # Get Coordinaets for Square Bound Box
    max_x = min(int(np.max(vert_2D[0,:])),imageW)
    min_x = max(int(np.min(vert_2D[0,:])),0)

    max_y = min(int(np.max(vert_2D[1,:])),imageH)
    min_y = max(int(np.min(vert_2D[1,:])),0)

    
    return min_x, min_y,max_x,max_y

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--trainDataLocation', type=str,help='Path to image test data',default='../workspace/gta_project/trainval')
    parser.add_argument('--imageDimSize', type=int,help='Dimensions of Image',default=120)

    args = parser.parse_args()
    trainDataLocation = args.trainDataLocation
    imageDimSize = args.imageDimSize

    # Specify Project Location
    trainDataSummaryFileLoc = Path('summarizedTrainImages.parquet')

    dim_train = (imageDimSize,imageDimSize)


    fileInfo = f'{imageDimSize}by{imageDimSize}SegmentedImages'

    # Read in Image Summary info
    df_image_info = pd.read_parquet(trainDataSummaryFileLoc)


    imageData = []
    y = []

    for i, row in df_image_info.iterrows():

        
        label = row['label']
        
        # Add Image Location
        imageLoc = row['filename']
        
        # Specify Image and Regions File to Look at
        imageFileLocation = Path(trainDataLocation) / Path(imageLoc)

        print(f"Processing Image File {i}: {imageLoc}")

       
        img = np.array(Image.open(imageFileLocation))
        # Get Image Shape
        imageH,imageW, _ = img.shape
         # Get Projection Bounding Boxes
        min_x, min_y, max_x, max_y = row[['min_x','min_y', 'max_x', 'max_y']]

        # Crop Full Sized Ground Truth Image
        img_cropped = img[min_y:max_y,min_x:max_x,:]

        # Resize
        img_cropped = cv2.resize(img_cropped, dim_train, interpolation = cv2.INTER_AREA)
        
        imageData.append(img_cropped) # Add Image
        y.append(label)


    imageData = np.array(imageData)
    np.save(f'XTrain{fileInfo}.npy',imageData)
    np.save(f'YTrain{fileInfo}.npy',y)
