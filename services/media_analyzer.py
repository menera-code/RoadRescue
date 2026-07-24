from ultralytics import YOLO
import cv2
import os
import json
from typing import List, Dict

# Load YOLO once at startup (pre-trained on COCO dataset)
model = YOLO('yolov8n.pt')  # 'n' for nano – fast and lightweight

def analyze_image(image_path: str) -> Dict:
    """Run YOLO on an image and return detected objects."""
    results = model(image_path)
    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            # Keep only relevant emergency-related objects (optional filter)
            detections.append({"label": label, "confidence": round(conf, 2)})
    return {"objects": detections, "count": len(detections)}

def analyze_video(video_path: str, frame_interval: int = 30) -> Dict:
    """Extract frames and analyze each."""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    all_detections = []
    temp_dir = "/tmp/video_frames"
    os.makedirs(temp_dir, exist_ok=True)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            temp_path = os.path.join(temp_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(temp_path, frame)
            result = analyze_image(temp_path)
            all_detections.extend(result["objects"])
            os.remove(temp_path)
        frame_count += 1
    cap.release()
    
    # Summarize unique objects
    from collections import Counter
    counter = Counter([d["label"] for d in all_detections])
    summary = [{"label": k, "count": v} for k, v in counter.most_common(10)]
    return {"summary": summary, "total_frames_analyzed": len(all_detections)}