import cv2
import pytesseract  # Make sure to import pytesseract
import numpy as np
import time
import torch
from torchvision import transforms

# Set up Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load YOLOv5 Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Your existing code continues...
