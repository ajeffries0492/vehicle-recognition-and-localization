import re # Regular Expression Library for Parsing Filenames
from pathlib import Path #Pathlib for dealing with file paths

import numpy as np # Linear Algebrar Library
import pandas as pd # Table Manipulation


import cv2 

def parse_filename(fullPathToFile,pattern='(\d*)_(\w*).(\w*)'):
    
    # Intialize DIctionary with File Info
    fileInfo = {}
    
    # Get Filename
    filename = str(fullPathToFile.name)
    
    # Add Directory and FileName to Parser
    folderName = str(fullPathToFile.parent.name)

    
    
    
    # Parse FileNames
    matches = re.findall(pattern,filename)
    
    if matches:
        sequenceIndex, fileType, fileExt = matches[0]
        
        fileInfo['baseFileName'] = sequenceIndex
        fileInfo['guid/image'] = f"{folderName}/{sequenceIndex}"

        fileInfo['filename'] = f"{folderName}/{filename}"
        
        
    return fileInfo
def rot(n):
    n = np.asarray(n).flatten()
    assert(n.size == 3)

    theta = np.linalg.norm(n)
    if theta:
        n /= theta
        K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])

        return np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    else:
        return np.identity(3)

def get_bbox(p0, p1):
    """
    Input:
    *   p0, p1
        (3)
        Corners of a bounding box represented in the body frame.

    Output:
    *   v
        (3, 8)
        Vertices of the bounding box represented in the body frame.
    *   e
        (2, 14)
        Edges of the bounding box. The first 2 edges indicate the `front` side
        of the box.
    """
    v = np.array([
        [p0[0], p0[0], p0[0], p0[0], p1[0], p1[0], p1[0], p1[0]],
        [p0[1], p0[1], p1[1], p1[1], p0[1], p0[1], p1[1], p1[1]],
        [p0[2], p1[2], p0[2], p1[2], p0[2], p1[2], p0[2], p1[2]]
    ])
    e = np.array([
        [2, 3, 0, 0, 3, 3, 0, 1, 2, 3, 4, 4, 7, 7],
        [7, 6, 1, 2, 1, 2, 4, 5, 6, 7, 5, 6, 5, 6]
    ], dtype=np.uint8)

    return v, e

def get_bbox_class(fullPathToBBoxFile):
    classes = (
    'Unknown', 'Compacts', 'Sedans', 'SUVs', 'Coupes',
    'Muscle', 'SportsClassics', 'Sports', 'Super', 'Motorcycles',
    'OffRoad', 'Industrial', 'Utility', 'Vans', 'Cycles',
    'Boats', 'Helicopters', 'Planes', 'Service', 'Emergency',
    'Military', 'Commercial', 'Trains'
    )
    # Intialize
    bboxClassInfo = {}
    
    # Read Bounding Box Data from File
    bboxes = np.fromfile(fullPathToBBoxFile, dtype=np.float32)
    bboxes = bboxes.reshape([-1, 11])
    
    # Read Projection Data from File
    proj = np.fromfile(str(fullPathToBBoxFile).replace('_bbox.bin', '_proj.bin'), dtype=np.float32)
    proj.resize([3, 4])

    nBoxes, boxSize = bboxes.shape

    for ibox,b in enumerate(bboxes):
        # Get Vehicle Type
        bboxClassInfo[f"BBox{ibox}_Class"] = classes[int(b[9])]
        
        
        R = rot(b[0:3])
        t = b[3:6]

        sz = b[6:9]
        vert_3D, edges = get_bbox(-sz / 2, sz / 2)
        vert_3D = R @ vert_3D + t[:, np.newaxis]

        vert_2D = proj @ np.vstack([vert_3D, np.ones(vert_3D.shape[1])])
        vert_2D = vert_2D / vert_2D[2, :]
        
        bboxClassInfo[f"BBox{ibox}_Edges"] = edges.tolist()
        
        bboxClassInfo[f"BBox{ibox}_Projection"] = vert_2D.tolist()
        
        
        
        
    return bboxClassInfo
def get_rectangular_box(vert_2D,imageW,imageH):


    # Get Coordinaets for Square Bound Box
    max_x = min(int(np.max(vert_2D[0,:])),imageW)
    min_x = max(int(np.min(vert_2D[0,:])),0)

    max_y = min(int(np.max(vert_2D[1,:])),imageH)
    min_y = max(int(np.min(vert_2D[1,:])),0)

    
    return min_x, min_y,max_x,max_y
def summarize_data(trainDataLocation,getBBOX = False,imageW=1914,imageH=1052):
    # Read in Labels
    pathToLabelFile = Path(trainDataLocation) / Path('labels.csv')

    if pathToLabelFile.is_file():
        df_label_info = pd.read_csv(pathToLabelFile)
        df_label_info =df_label_info.set_index('guid/image')
    ## Loop Through List of train data images

    fileInfos = []
    for fileLocation in Path(trainDataLocation).glob('*/**/*.jpg'):
        # Parse Image Filename
        fileInfo = parse_filename(fileLocation)

        if pathToLabelFile.is_file():
            # Grab Label Info
            fileInfo['label']=df_label_info.loc[fileInfo['guid/image'],'label']

        # Look at All Bin Files if Available
        pathToFile = Path(fileLocation).parent
        if getBBOX:
            fullPathToFile = pathToFile / Path(f"{fileInfo['baseFileName']}_bbox.bin")
            # Ensure That Bin File Exists
            if fullPathToFile.is_file():

                
                # Get Boundng Box Class Info
                bboxClassInfo = get_bbox_class(fullPathToFile)


                vert_2D = bboxClassInfo[f"BBox{0}_Projection"]

                vert_2D = np.array([v for v in vert_2D])

                min_x, min_y,max_x,max_y = get_rectangular_box(vert_2D,imageW,imageH)
                boxInfo = {'class': bboxClassInfo['BBox0_Class'],
                'min_x': min_x,
                'min_y': min_y,
                'max_x': max_x,
                'max_y': max_y}
                # Update Class Info
                fileInfo.update(boxInfo)

        fileInfos.append(fileInfo)

    df_image_info = pd.DataFrame(fileInfos)
    

    # df_image_info.sort_values(by=['guid/image'],ignore_index=True,inplace=True)
    
    return df_image_info

def get_image(fileLocation,grayed=False):
    
    
    # Read in Image
    img = cv2.imread(str(fileLocation))
    
    img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    if grayed:
        # Convert Image to GrayScale
        img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    
        
    return img
def rescale_image(image,scale_percent=None,dim=None):
    
    scaledImage = image
    # Scale if Scale Percent is different
    if scale_percent is not None:
        # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        
        # Ressized Dims
        resizedDims = (width, height)

        # resize image
        scaledImage = cv2.resize(image, resizedDims, interpolation = cv2.INTER_AREA)
    elif dim is not None:
        # resize image
        scaledImage = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        
    return scaledImage

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--trainDataLocation', type=str,help='Path to image train data',default='workspace/gta_project/trainval')
    parser.add_argument('--testDataLocation', type=str,help='Path to image test data',default='workspace/gta_project/test')

    args = parser.parse_args()
    testDataLocation = args.testDataLocation
    trainDataLocation = args.trainDataLocation

    
    # Save DataFrame Summarizing Parquet Train File
    df_train_info = summarize_data(trainDataLocation,getBBOX=True)
    df_train_info.to_parquet(f"summarizedTrainInfo.parquet")

    # Save DataFrame Summarizing Parquet Test File
    df_test_info = summarize_data(testDataLocation,getBBOX=False)
    df_test_info.to_parquet(f"summarizedTestInfo.parquet")