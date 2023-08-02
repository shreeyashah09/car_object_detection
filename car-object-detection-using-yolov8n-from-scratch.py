#!/usr/bin/env python
# coding: utf-8

# # Car Object Detection Using YoloV8

# ## Installing YoloV8 from Ultralytics

# In[1]:


# install yolov8
get_ipython().system('pip install ultralytics')


# In[3]:


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
from IPython import display
display.clear_output()
get_ipython().system('yolo mode=checks')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


import ultralytics
ultralytics.checks()


# ## Setting Parameters

# In[5]:


DIR = "/kaggle/working/datasets/cars/"
IMAGES = DIR +"images/"
LABELS = DIR +"labels/"

TRAIN = "/kaggle/input/car-object-detection/data/training_images"
TEST = "/kaggle/input/car-object-detection/data/testing_images"


# In[6]:


df = pd.read_csv("/kaggle/input/car-object-detection/data/train_solution_bounding_boxes (1).csv")
df.head()


# In[7]:


get_ipython().system(' pwd')


# ## Setting dataset

# In[8]:


files = list(df.image.unique())

files_train, files_valid = train_test_split(files, test_size = 0.2)


# In[9]:


# make directories
os.makedirs(IMAGES+"train", exist_ok=True)
os.makedirs(LABELS+"train", exist_ok=True)
os.makedirs(IMAGES+"valid", exist_ok=True)
os.makedirs(LABELS+"valid", exist_ok=True)


# In[10]:


train_filename = set(files_train)
valid_filename = set(files_valid)
for file in glob.glob(TRAIN+"/*"):
    fname =os.path.basename(file)
    if fname in train_filename:
        sh.copy(file, IMAGES+"train")
    elif fname in valid_filename:
        sh.copy(file, IMAGES+"valid")


# In[11]:


get_ipython().run_line_magic('cd', '/kaggle/working/datasets/cars')


# In[12]:


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


# ## Creating yaml file

# In[13]:


get_ipython().run_cell_magic('writefile', 'dataset.yaml', "# Path\npath: ./cars\ntrain: images/train\nval: images/valid\n\n# Class\nnc: 1\n# name of class    \nnames: ['car']")


# In[14]:


get_ipython().system('ls /kaggle/working/datasets/cars')


# ## Training the Model from scratch

# In[15]:


model = YOLO()
model.train(data="/kaggle/working/datasets/cars/dataset.yaml", epochs=50) # train the model

#model = YOLO("yolov8m.pt") #load a pretrained YOLOv8m model


# In[16]:


get_ipython().system('ls /kaggle/working/datasets/cars/runs/detect/train')


# In[20]:


Image(filename=f"/kaggle/working/datasets/cars/runs/detect/train/results.png", width=1200)


# In[21]:


Image(filename=f"/kaggle/working/datasets/cars/runs/detect/train/confusion_matrix.png", width=800)


# ## Validate the model

# In[22]:


get_ipython().system('yolo task=detect mode=val model=/kaggle/working/datasets/cars/runs/detect/train/weights/best.pt data=dataset.yaml')


# In[23]:


get_ipython().system('ls /kaggle/working/datasets/cars/runs/detect/val')


# ## Model Prediction on Validation Batch

# In[26]:


Image(filename=f"/kaggle/working/datasets/cars/runs/detect/val/val_batch2_pred.jpg", width=1200)


# ## Prediction on the Custom Model

# In[32]:


#Predicting data using a custom model
get_ipython().system('yolo task=detect mode=predict model=/kaggle/working/datasets/cars/runs/detect/train/weights/best.pt conf=0.5 source=/kaggle/input/car-object-detection/data/testing_images')


# In[38]:


Image(filename=f"/kaggle/working/datasets/cars/runs/detect/predict/vid_5_26660.jpg", width=600)


# In[34]:


Image(filename=f"/kaggle/working/datasets/cars/runs/detect/predict/vid_5_31040.jpg", width=600)


# In[35]:


Image(filename=f"/kaggle/working/datasets/cars/runs/detect/predict/vid_5_27480.jpg", width=600)


# In[39]:


#Predicting data using a custom model
get_ipython().system('yolo task=detect mode=predict model=/kaggle/working/datasets/cars/runs/detect/train/weights/best.pt conf=0.5 source=/kaggle/input/car-videos/car_video1.mp4')


# In[40]:


from IPython.display import Video

avi_video_path = 'kaggle/working/datasets/cars/runs/detect/predict4/car_video1.avi'
Video(avi_video_path, embed=True)


# In[43]:


#Predicting data using a custom model
get_ipython().system('yolo task=detect mode=predict model=/kaggle/working/datasets/cars/runs/detect/train/weights/best.pt conf=0.5 source=/kaggle/input/car-videos/car_video2.webm')


# In[44]:


from IPython.display import Video

avi_video_path = 'kaggle/working/datasets/cars/runs/detect/predict5/car_video2.avi'
Video(avi_video_path, embed=True)


# In[45]:


#Predicting data using a custom model
get_ipython().system('yolo task=detect mode=predict model=/kaggle/working/datasets/cars/runs/detect/train/weights/best.pt conf=0.5 source=/kaggle/input/carvideo2/carvideo2.mp4')


# In[46]:


from IPython.display import Video

avi_video_path = 'kaggle/working/datasets/cars/runs/detect/predict6/carvideo2.avi'
Video(avi_video_path, embed=True)


# In[ ]:




