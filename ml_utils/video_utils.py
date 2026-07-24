# backend/ml_utils/video_utils.py
import cv2
from PIL import Image
import numpy as np

def extract_frames(video_path, num_frames=8, size=(224,224)):
    """
    Extract evenly spaced frames from a video.
    Returns a list of PIL Images.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []
    # Calculate step to get exactly num_frames (or less)
    step = max(1, total_frames // num_frames)
    frames = []
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb).resize(size, Image.LANCZOS)
            frames.append(pil_img)
        if len(frames) >= num_frames:
            break
    cap.release()
    return frames