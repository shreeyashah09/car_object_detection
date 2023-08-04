import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch

# Load the YOLOv8 model with the best weights
def load_yolov8_model():
    model = YOLO('best.pt')
    return model 

def detect_objects_yolov8(image, model):
    # YOLOv8 object detection code here
    results = model.predict(image)
    return results

# Streamlit app
def main():
    st.title("Car Object Detection App")
    st.write("Upload an image and let YOLOv8 detect objects in it!")

    # File upload
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert the image to OpenCV format
        image_cv = np.array(image)

        # Load YOLOv8 model (call the loading function you have)
        model = load_yolov8_model()

        # Detect objects using YOLOv8
        results = detect_objects_yolov8(image_cv, model)

        # Draw bounding boxes on the image
        detected_image = image_cv.copy()
        for result in results.pred:
            for box in result[:, :4]:
                x_min, y_min, x_max, y_max = box
                color = (0, 255, 0)  # Green color for bounding boxes
                thickness = 2
                cv2.rectangle(detected_image, (x_min, y_min), (x_max, y_max), color, thickness)

        # Convert back to PIL format for displaying
        detected_image_pil = Image.fromarray(detected_image)

        # Display the detected image
        st.image(detected_image, caption="Detected Objects", use_column_width=True)

if __name__ == "__main__":
    main()