import cv2
import pytesseract
import numpy as np
import time
import torch
import tkinter as tk
from tkinter import filedialog, messagebox
from threading import Thread
from PIL import Image, ImageTk  # To handle the logo image
import sys
import pandas as pd

# Set up Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update if needed

# Redirect stderr to avoid NoneType error
class DummyWriter:
    def write(self, message):
        pass
    def flush(self):
        pass

sys.stderr = DummyWriter()

# Load YOLOv5 Model
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
except Exception as e:
    print("Error loading YOLOv5 model:", e)
    model = None

# Initialize parameters
fps = 30  # Default frames per second
MAX_SPEED = 80  # Maximum speed limit for capping
running = True  # Flag to control video processing

def detect_vehicles(frame):
    if model is not None:
        results = model(frame)
        detections = results.pandas().xyxy[0]  # Bounding boxes with confidence, class, etc.
        return detections
    return pd.DataFrame()  # Return an empty DataFrame if the model is not loaded
    
import cv2
import pytesseract

def recognize_plate(frame, bbox):
    """Recognizes a vehicle plate number from a given image frame and bounding box.

    Args:
        frame: The input image frame.
        bbox: The bounding box coordinates (x1, y1, x2, y2) of the plate region.

    Returns:
        The recognized plate number string, or None if no plate was detected.
    """

    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    plate_img = frame[y1:y2, x1:x2]

    # Preprocess the plate image
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply morphological operations to enhance plate contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find contours of plate characters
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on size and shape
    valid_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > h and w > 10 and h > 10:
            valid_contours.append(cnt)

    # Sort contours left to right
    valid_contours = sorted(valid_contours, key=lambda c: cv2.boundingRect(c)[0])

    # Extract and recognize characters from each contour
    plate_text = ""
    for cnt in valid_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        char_img = thresh[y:y+h, x:x+w]
        char_text = pytesseract.image_to_string(char_img, config='--psm 6').strip()
        plate_text += char_text

    return plate_text.strip()


def estimate_speed(positions, timestamps):
    if len(positions) < 2:
        return 0

    delta_x = positions[-1][0] - positions[-2][0]
    delta_y = positions[-1][1] - positions[-2][1]
    distance = np.sqrt(delta_x**2 + delta_y**2)
    time_diff = timestamps[-1] - timestamps[-2]

    speed = distance / time_diff if time_diff > 0 else 0
    adjusted_distance_per_frame = 0.01
    speed_kmh = speed * adjusted_distance_per_frame * fps * 3.6  # Convert to km/h

    return min(speed_kmh, MAX_SPEED)



def detect_color(frame, bbox):
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    vehicle_img = frame[y1:y2, x1:x2]
    hsv_img = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2HSV)

    # Improved color detection with wider ranges and additional conditions
    COLOR_BOUNDS = {
        'Red': [(np.array([0, 100, 100]), np.array([30, 255, 255])),
                 (np.array([160, 100, 100]), np.array([180, 255, 255]))],
        'Green': [(np.array([30, 50, 50]), np.array([90, 255, 255]))],
        'Blue': [(np.array([90, 100, 100]), np.array([140, 255, 255]))],
        'Yellow': [(np.array([20, 100, 100]), np.array([30, 255, 255]))],
        'White': [(np.array([0, 0, 200]), np.array([180, 25, 255]))],
        'Black': [(np.array([0, 0, 0]), np.array([180, 255, 50]))]
    }

    detected_color = 'Unknown'
    for color_name, bounds in COLOR_BOUNDS.items():
        for lower_bound, upper_bound in bounds:
            mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
            # Check for minimum number of pixels to avoid false positives
            if cv2.countNonZero(mask) > 0.1 * vehicle_img.shape[0] * vehicle_img.shape[1]:
                detected_color = color_name
                break
        if detected_color != 'Unknown':
            break

    return detected_color



def process_video(video_path):
    global running
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video file.")
        return
    
    positions = []
    timestamps = []
    vehicle_count = 0

    while cap.isOpened() and running:
        ret, frame = cap.read()
        if not ret:
            break
        
        height, width = frame.shape[:2]
        screen_width = 1280
        screen_height = int(screen_width * height / width)
        resized_frame = cv2.resize(frame, (screen_width, screen_height))

        detections = detect_vehicles(frame)

        for _, detection in detections.iterrows():
            if detection['confidence'] > 0.4 and detection['name'] == 'car':
                vehicle_count += 1
                x1 = int(detection['xmin'] * screen_width / width)
                y1 = int(detection['ymin'] * screen_height / height)
                x2 = int(detection['xmax'] * screen_width / width)
                y2 = int(detection['ymax'] * screen_height / height)

                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                positions.append((center_x, center_y))
                timestamps.append(time.time())
                
                if len(positions) > 1:
                    speed = estimate_speed(positions, timestamps)
                    cv2.putText(resized_frame, f"Speed: {speed:.2f} km/h", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                plate_text = recognize_plate(resized_frame, [x1, y1, x2, y2])
                if plate_text:
                    cv2.putText(resized_frame, f"Plate: {plate_text}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                vehicle_color = detect_color(resized_frame, [x1, y1, x2, y2])
                cv2.putText(resized_frame, f"Color: {vehicle_color}", (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.putText(resized_frame, f"Vehicles Detected: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow("AutoEye Vehicle Detection", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def close_app():
    global running
    running = False
    cv2.destroyAllWindows()
    messagebox.showinfo("Info", "Application Closed")

def select_video():
    video_path = filedialog.askopenfilename(title="Select a Video File", filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    if video_path:
        Thread(target=process_video, args=(video_path,)).start()
    else:
        messagebox.showwarning("No File Selected", "Please select a valid video file.")

def create_app():
    root = tk.Tk()
    root.title("AutoEye")

    # Set background color to grey
    root.configure(bg="grey")

    # Load the logo image
    logo = Image.open("logo.jpeg")  # Update the path to the logo
    logo = logo.resize((500, 600))  # Resize the logo if needed
    logo_image = ImageTk.PhotoImage(logo)

    # Create a label for the logo
    logo_label = tk.Label(root, image=logo_image, bg="grey")
    logo_label.pack(pady=10)

    select_button = tk.Button(root, text="Select Video File", command=select_video)
    select_button.pack(pady=20)

    exit_button = tk.Button(root, text="Close Output", command=close_app)
    exit_button.pack(pady=10)

    root.mainloop()

if __name__ == '__main__':
    create_app()
