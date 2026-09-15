# detector_utils.py - Frame Extraction & Computer Vision Helpers[cite: 1]
import cv2
import numpy as np

def extract_face_region(frame):
    """crop vifeo"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape
    crop_h, crop_w = int(h * 0.6), int(w * 0.6)
    start_y = (h - crop_h) // 4
    start_x = (w - crop_w) // 2
    return rgb_frame[start_y:start_y+crop_h, start_x:start_x+crop_w]

def analyze_texture_artifacts(image_np):
    """check smoothness in ai generated video]"""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var