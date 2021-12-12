import os
import sys
import io
import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf
from sklearn.model_selection import train_test_split
from pathlib import Path

# sys.path.append("../models/research")

import contextlib2
from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from collections import namedtuple, OrderedDict

def split(df, group):
    data = namedtuple("data", ["filename", "object"])
    gb = df.groupby(group)
    return [
        data(filename, gb.get_group(x))
        for filename, x in zip(gb.groups.keys(), gb.groups)
    ]

def class_text_to_int(row_label,label_map):
    label_map_dict = label_map_util.get_label_map_dict(label_map)
    return label_map_dict[row_label]

def create_tf_record(group, path,label_map,column_label_name ='label'):
   

    with open(os.path.join(path, "{}".format(group.filename)), "rb") as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode("utf8")
    image_format = b"jpg"

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for _,row in group.object.iterrows():
        xmins.append(row["min_x"] / width)
        xmaxs.append(row["max_x"] / width)
        ymins.append(row["min_y"] / height)
        ymaxs.append(row["max_y"] / height)
        classes_text.append(row[column_label_name].encode("utf8"))
        classes.append(class_text_to_int(row[column_label_name],label_map))

    tf_sample = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/height": dataset_util.int64_feature(height),
                "image/width": dataset_util.int64_feature(width),
                "image/filename": dataset_util.bytes_feature(filename),
                "image/source_id": dataset_util.bytes_feature(filename),
                "image/encoded": dataset_util.bytes_feature(encoded_jpg),
                "image/format": dataset_util.bytes_feature(image_format),
                "image/object/bbox/xmin": dataset_util.float_list_feature(xmins),
                "image/object/bbox/xmax": dataset_util.float_list_feature(xmaxs),
                "image/object/bbox/ymin": dataset_util.float_list_feature(ymins),
                "image/object/bbox/ymax": dataset_util.float_list_feature(ymaxs),
                "image/object/class/text": dataset_util.bytes_list_feature(classes_text),
                "image/object/class/label": dataset_util.int64_list_feature(classes)
            }
        )
    )
    return tf_sample
def write_tf_records(input_image_path,label_map,grouped,output_records_path,num_shards=10,column_label_name='class2'):
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(tf_record_close_stack, output_records_path, num_shards)

        for index,row in enumerate(grouped):


            tf_record = create_tf_record(row, input_image_path,label_map,column_label_name=column_label_name)
            output_shard_index = index % num_shards
            output_tfrecords[output_shard_index].write(tf_record.SerializeToString())
    
    print("Successfully created the TFRecords: {}".format(output_records_path))

def get_label_map(label_map_path):
    label_map = label_map_util.load_labelmap(label_map_path)
    
    
    return label_map

def create_pbtxt(label_map_path,classes_names):
     # Create the `label_map.pbtxt` file
    pbtxt_content = ""
    for i, class_name in enumerate(classes_names):
        pbtxt_content = (
            pbtxt_content
            + "item {{\n    id: {0}\n    name: '{1}'\n}}\n\n".format(
                i + 1, class_name
            )
        )
    pbtxt_content = pbtxt_content.strip()
    
    with open(label_map_path, "w") as f:
        f.write(pbtxt_content)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_summary', type=str,help='Path to image train data summary file',default='summarizedTrainInfo.parquet')
    parser.add_argument('--updated_label_names', type=str,help='Path to updated labels CSV',default='classes2.csv')
    
    parser.add_argument('--train_input_data_path', type=str,help='Path to train data',default='workspace/gta_project/trainval')
    parser.add_argument('--train_output_data_path', type=str,help='Path to train output data',default='workspace/gta_project/annotations')
    parser.add_argument('--label_map_path', type=str,help='Path to save pbtxt label map',default='workspace/gta_project/annotations/label_map.pbtxt')
    parser.add_argument('--column_label_name',type=str,help='',default='class2')
    parser.add_argument('--num_shards',type=int,help='',default=10)
    parser.add_argument('--test_split',type=float,help='',default=0.1)

    args = parser.parse_args()
    train_data_summary = args.train_data_summary
    updated_label_names = args.updated_label_names
    train_input_data_path = args.train_input_data_path
    train_output_data_path = args.train_output_data_path
    label_map_path = args.label_map_path
    column_label_name = args.column_label_name
    num_shards = args.num_shards
    test_split = args.test_split
   

    # Load in Pandas Summary DataFrame
    df_train_image_info = pd.read_parquet(train_data_summary)

    df_labels = pd.read_csv(updated_label_names)
    labelConverter = dict(tuple(df_labels.loc[:,['label','class_name2']].values))

    

    df_train_image_info['class2'] = df_train_image_info['label'].apply(lambda x: labelConverter[x])    
    # df_train_image_info['class2'] = 'vehicle'

    # Create Label Map
    class_names = list(labelConverter.values())
    print(class_names)
    
    create_pbtxt(label_map_path,class_names)
    # Get Label Map
    label_map = get_label_map(label_map_path)

    # Group Files
    grouped = split(df_train_image_info, "filename")

    # Divide Train and Test Data
    XTrain,XTest = train_test_split(grouped,test_size=test_split)
    # Write Records for Training Data
    output_records_path = Path(train_output_data_path) / Path("train_dataset.records")
    write_tf_records(train_input_data_path,label_map,XTrain,output_records_path,num_shards=10,column_label_name=column_label_name)

    # Write Records for Test Data
    output_records_path = Path(train_output_data_path) / Path("test_dataset.records")
    write_tf_records(train_input_data_path,label_map,XTest,output_records_path,num_shards=10,column_label_name=column_label_name)

    
