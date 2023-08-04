from ultralytics import YOLO
import streamlit as st
import cv2
# import pafy

from django.conf import settings


def load_model(model_path):
    model = YOLO(model_path)
    return model