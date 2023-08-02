from ultralytics import YOLO
import os, time, random
import numpy as np
import pandas as pd
import cv2, torch
from tqdm.auto import tqdm
import shutil as sh
import yaml
import glob
from sklearn.model_selection import train_test_split

from IPython.display import Image, clear_output, display
import matplotlib.pyplot as plt
from IPython import display, get_ipython
display.clear_output()

import ultralytics
ultralytics.checks()

# Setting Parameters
DIR = "/kaggle/working/datasets/cars/"
IMAGES = DIR +"images/"
LABELS = DIR +"labels/"

TRAIN = "/kaggle/input/car-object-detection/data/training_images"
TEST = "/kaggle/input/car-object-detection/data/testing_images"

df = pd.read_csv("/kaggle/input/car-object-detection/data/train_solution_bounding_boxes (1).csv")
df.head()

# Setting dataset
files = list(df.image.unique())
files_train, files_valid = train_test_split(files, test_size = 0.2)

# make directories
os.makedirs(IMAGES+"train", exist_ok=True)
os.makedirs(LABELS+"train", exist_ok=True)
os.makedirs(IMAGES+"valid", exist_ok=True)
os.makedirs(LABELS+"valid", exist_ok=True)


train_filename = set(files_train)
valid_filename = set(files_valid)
for file in glob.glob(TRAIN+"/*"):
    fname =os.path.basename(file)
    if fname in train_filename:
        sh.copy(file, IMAGES+"train")
    elif fname in valid_filename:
        sh.copy(file, IMAGES+"valid")


for _, row in df.iterrows():    
    image_file = row['image']
    class_id = "0"
    x = row['xmin']
    y = row['ymin']
    width = row['xmax'] - row['xmin']
    height = row['ymax'] - row['ymin']

    x_center = x + (width / 2)
    y_center = y + (height / 2)
    x_center /= 676
    y_center /= 380
    width /= 676
    height /= 380

    if image_file in train_filename:   
        annotation_file = os.path.join(LABELS) + "train/" + image_file.replace('.jpg', '.txt')
    else:
        annotation_file = os.path.join(LABELS) + "valid/" + image_file.replace('.jpg', '.txt')
        
    with open(annotation_file, 'a') as ann_file:
        ann_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# Training the Model from scratch

model = YOLO()
model.train(data="/kaggle/working/datasets/cars/dataset.yaml", epochs=50) # train the model

# model = YOLO("yolov8m.pt") #load a pretrained YOLOv8m model

Image(filename=f"/kaggle/working/datasets/cars/runs/detect/train/results.png", width=1200)

Image(filename=f"/kaggle/working/datasets/cars/runs/detect/train/confusion_matrix.png", width=800)


# Validate the model
get_ipython().system('yolo task=detect mode=val model=/kaggle/working/datasets/cars/runs/detect/train/weights/best.pt data=dataset.yaml')


# Model Prediction on Validation Batch
Image(filename=f"/kaggle/working/datasets/cars/runs/detect/val/val_batch2_pred.jpg", width=1200)

#Predicting data using a custom model
get_ipython().system('yolo task=detect mode=predict model=/kaggle/working/datasets/cars/runs/detect/train/weights/best.pt conf=0.5 source=/kaggle/input/car-object-detection/data/testing_images')


Image(filename=f"/kaggle/working/datasets/cars/runs/detect/predict/vid_5_26660.jpg", width=600)

Image(filename=f"/kaggle/working/datasets/cars/runs/detect/predict/vid_5_31040.jpg", width=600)

Image(filename=f"/kaggle/working/datasets/cars/runs/detect/predict/vid_5_27480.jpg", width=600)


#Predicting data using a custom model
get_ipython().system('yolo task=detect mode=predict model=/kaggle/working/datasets/cars/runs/detect/train/weights/best.pt conf=0.5 source=/kaggle/input/car-videos/car_video1.mp4')


#Predicting data using a custom model
get_ipython().system('yolo task=detect mode=predict model=/kaggle/working/datasets/cars/runs/detect/train/weights/best.pt conf=0.5 source=/kaggle/input/car-videos/car_video2.webm')


from IPython.display import Video
avi_video_path = 'kaggle/working/datasets/cars/runs/detect/predict5/car_video2.avi'
Video(avi_video_path, embed=True)

