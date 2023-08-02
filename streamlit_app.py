import streamlit as st
from ultralytics import YOLO
from PIL import Image

model_path = 'best.pt'  # Replace with the path to your YOLOv8 model weights 
yolo = YOLO(model_path)

# C:\Users\ks010\Desktop\GBC\13. Deep Learning II\Project - Car Object Detection\car_object_detection\best.pt
'''
# Setting page layout
st.set_page_config(
    page_title=" Car Object Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Creating sidebar
with st.sidebar:
    st.header("Image Config")
    source_img = st.file_uploader("Upload an image...", type=("jpg", "jpeg", "png", "bmp", "webp"))
    confidence = float(st.slider("Select Model Confidence", 25, 100, 40)) / 100


if source_img is not None:
    st.title(" Car Object Detection")
    # Load the uploaded image
    img = Image.open(source_img)

    # Run object detection on the image
    results = yolo(source_img, conf=confidence)

    # Display the original image with bounding boxes around detected objects
    st.image(results.render()[0])
'''