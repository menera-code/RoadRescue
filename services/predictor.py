"""
ML Prediction Service – Road Rescue Edition
- Always predicts incident type = "Accident"
- Extracts vehicle types from text description
- Detects vehicles from images/videos using YOLO (pre-trained on COCO)
- Severity classification remains (zero-shot)
"""
from ultralytics import YOLO
from transformers import pipeline
import cv2
import os
import json
from collections import Counter
from typing import Dict, List

# ---------- TEXT CLASSIFIER (zero-shot) ----------
_text_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

SEVERITY_LEVELS = ["low", "medium", "high", "critical"]

# Expanded vehicle keywords (English + common local terms)
VEHICLE_KEYWORDS = [
    "car", "cars", "sedan", "suv", "van", "pickup", "hatchback",
    "truck", "trucks", "lorry", "dump truck",
    "motorcycle", "motorbike", "bike", "scooter", "moped",
    "tricycle", "trike", "pedicab",
    "bus", "minibus", "coaster",
    "bicycle", "bike",
    "jeepney", "jeep",
    "trailer", "semi-trailer", "heavy vehicle",
    "ambulance", "fire truck", "police car"
]

def extract_vehicles_from_text(text: str) -> List[str]:
    """Return list of unique vehicle types mentioned in the description."""
    text_lower = text.lower()
    found = []
    for vehicle in VEHICLE_KEYWORDS:
        if vehicle in text_lower:
            # Capitalize first letter for display
            found.append(vehicle.title())
    # Remove duplicates while preserving order
    unique = []
    for v in found:
        if v not in unique:
            unique.append(v)
    return unique

def predict_text(description: str) -> Dict:
    """
    Always returns incident_type = "Accident".
    Extracts mentioned vehicles and predicts severity.
    """
    severity_result = _text_classifier(description, SEVERITY_LEVELS)
    vehicles = extract_vehicles_from_text(description)
    return {
        "incident_type": "Accident",          # Always road accident
        "type_confidence": 1.0,
        "severity": severity_result["labels"][0],
        "severity_confidence": round(severity_result["scores"][0], 2),
        "all_severity_scores": {
            label: round(score, 2)
            for label, score in zip(severity_result["labels"], severity_result["scores"])
        },
        "mentioned_vehicles": vehicles       # New field for UI
    }

# ---------- IMAGE / VIDEO DETECTION (YOLO) ----------
_yolo = YOLO('yolov8n.pt')  # pre-trained on COCO

# Map YOLO COCO class IDs to vehicle types (only relevant ones)
VEHICLE_CLASSES = {
    2: "car",          # car
    3: "motorcycle",   # motorcycle
    5: "bus",          # bus
    7: "truck",        # truck
    1: "bicycle",      # bicycle
    # Note: Class 4 = airplane (ignore), 6 = train (rare)
}

def analyze_image(image_path: str) -> Dict:
    """
    Detect objects in a single image.
    Returns all objects + a breakdown of detected vehicles.
    """
    results = _yolo(image_path)
    detections = []
    vehicles_detected = Counter()
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = _yolo.names[cls]
            conf = float(box.conf[0])
            detections.append({"label": label, "confidence": round(conf, 2)})
            if cls in VEHICLE_CLASSES:
                vehicles_detected[VEHICLE_CLASSES[cls]] += 1
    return {
        "objects": detections,
        "count": len(detections),
        "vehicles": dict(vehicles_detected)   # e.g. {"car": 2, "motorcycle": 1}
    }

def analyze_video(video_path: str, frame_interval: int = 30) -> Dict:
    """
    Extract frames and analyze each, aggregating vehicle detections.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    all_vehicles = Counter()
    temp_dir = "/tmp/video_frames"
    os.makedirs(temp_dir, exist_ok=True)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            temp_path = os.path.join(temp_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(temp_path, frame)
            res = analyze_image(temp_path)
            for vehicle, count in res.get("vehicles", {}).items():
                all_vehicles[vehicle] += count
            os.remove(temp_path)
        frame_count += 1
    cap.release()
    
    # Summarize detected vehicles (most common first)
    summary = [{"label": k, "count": v} for k, v in all_vehicles.most_common(10)]
    return {
        "summary": summary,
        "total_frames_analyzed": frame_count // frame_interval,
        "vehicles": dict(all_vehicles)
    }