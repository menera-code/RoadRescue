import cv2
import numpy as np
import os
from typing import List, Dict, Any

# Try to import YOLO, but fallback if not available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO not available. Install with: pip install ultralytics")

class EnhancedImageAnalyzer:
    def __init__(self, model_path="ml_models/saved_models/yolo/"):
        self.model_path = model_path
        os.makedirs(model_path, exist_ok=True)
        
        # Initialize YOLO model if available
        self.model = None
        if YOLO_AVAILABLE:
            try:
                # Try to load custom-trained model
                self.model = YOLO(os.path.join(model_path, "best.pt"))
                print("✓ Loaded custom YOLO model")
            except:
                try:
                    # Use pre-trained YOLOv8
                    self.model = YOLO("yolov8n.pt")
                    print("✓ Loaded pre-trained YOLOv8 model")
                except Exception as e:
                    print(f"⚠️ Could not load YOLO: {e}")
                    self.model = None
        else:
            print("⚠️ YOLO not installed, using basic analysis")
        
        # Incident-related classes (COCO classes + custom)
        self.emergency_classes = {
            'person': {'severity': 1, 'risk': 'low'},
            'car': {'severity': 2, 'risk': 'medium'},
            'truck': {'severity': 2, 'risk': 'medium'},
            'bus': {'severity': 2, 'risk': 'medium'},
            'motorcycle': {'severity': 2, 'risk': 'medium'},
            'fire': {'severity': 4, 'risk': 'critical'},
            'smoke': {'severity': 3, 'risk': 'high'},
            'accident': {'severity': 3, 'risk': 'high'},
            'damage': {'severity': 3, 'risk': 'high'},
        }
        
        # Color detection ranges for emergency signals
        self.color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255]),  # Fire, danger
            'orange': ([10, 100, 100], [25, 255, 255]),  # Warning
            'blue': ([90, 50, 50], [130, 255, 255]),  # Emergency vehicles
            'yellow': ([25, 100, 100], [35, 255, 255])  # Caution
        }
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Enhanced image analysis with YOLO or fallback"""
        try:
            if not os.path.exists(image_path):
                return self.default_response()
            
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return self.default_response()
            
            height, width = img.shape[:2]
            
            # YOLO Object Detection if available
            if self.model:
                yolo_results = self.yolo_detection(img)
            else:
                yolo_results = self.basic_detection(img)
            
            # Color analysis for emergency signals
            color_analysis = self.analyze_emergency_colors(img)
            
            # Scene complexity analysis
            scene_complexity = self.analyze_scene_complexity(img)
            
            # Determine severity based on detections
            severity, severity_score = self.determine_severity(yolo_results, color_analysis)
            
            # Calculate confidence
            confidence = 0.5 + (severity_score * 0.1) + (len(yolo_results['objects']) * 0.05)
            confidence = min(confidence, 0.95)
            
            return {
                "success": True,
                "severity": severity,
                "severity_score": severity_score,
                "confidence": confidence,
                "detected_objects": yolo_results['objects'],
                "object_count": yolo_results['object_count'],
                "color_analysis": color_analysis,
                "scene_complexity": scene_complexity,
                "image_size": f"{width}x{height}",
                "analysis_method": "yolo" if self.model else "basic"
            }
            
        except Exception as e:
            print(f"Enhanced image analysis error: {e}")
            # Return default response instead of trying to import from ml_models
            return self.default_response()
    
    def yolo_detection(self, img) -> Dict[str, Any]:
        """Run YOLO object detection"""
        try:
            # Run inference
            results = self.model(img, conf=0.25)
            
            detected_objects = []
            emergency_score = 0
            person_count = 0
            vehicle_count = 0
            
            for result in results:
                if hasattr(result, 'boxes'):
                    boxes = result.boxes.cpu().numpy()
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        bbox = box.xyxy[0].astype(int)
                        
                        # Get class name
                        class_name = self.model.names[cls_id]
                        
                        # Map to emergency classes
                        emergency_info = self.emergency_classes.get(class_name, {
                            'severity': 1,
                            'risk': 'low'
                        })
                        
                        detected_objects.append({
                            "class": class_name,
                            "confidence": confidence,
                            "bbox": bbox.tolist(),
                            "severity": emergency_info['severity'],
                            "risk": emergency_info['risk']
                        })
                        
                        # Count specific objects
                        if class_name == 'person':
                            person_count += 1
                        elif class_name in ['car', 'truck', 'bus', 'motorcycle']:
                            vehicle_count += 1
                        
                        # Add to emergency score
                        emergency_score += emergency_info['severity'] * confidence
            
            return {
                "objects": detected_objects,
                "object_count": len(detected_objects),
                "person_count": person_count,
                "vehicle_count": vehicle_count,
                "emergency_score": emergency_score / max(len(detected_objects), 1)
            }
            
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return self.basic_detection(img)
    
    def basic_detection(self, img) -> Dict[str, Any]:
        """Basic object detection when YOLO is not available"""
        try:
            # Simple contour detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detected_objects = []
            for i, contour in enumerate(contours[:10]):
                area = cv2.contourArea(contour)
                if area > 500:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Simple classification
                    aspect_ratio = w / h if h > 0 else 1
                    if aspect_ratio > 1.5:
                        obj_type = "vehicle_like"
                        severity = 2
                    elif aspect_ratio < 0.5:
                        obj_type = "person_like"
                        severity = 1
                    else:
                        obj_type = "object"
                        severity = 1
                    
                    detected_objects.append({
                        "class": obj_type,
                        "confidence": 0.6,
                        "bbox": [x, y, w, h],
                        "severity": severity,
                        "risk": "low"
                    })
            
            return {
                "objects": detected_objects,
                "object_count": len(detected_objects),
                "person_count": sum(1 for obj in detected_objects if obj["class"] == "person_like"),
                "vehicle_count": sum(1 for obj in detected_objects if obj["class"] == "vehicle_like"),
                "emergency_score": len(detected_objects) * 0.3
            }
            
        except Exception as e:
            print(f"Basic detection error: {e}")
            return {"objects": [], "object_count": 0, "emergency_score": 0}
    
    def analyze_emergency_colors(self, img) -> Dict[str, Any]:
        """Analyze emergency-related colors in image"""
        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            height, width = img.shape[:2]
            
            color_detections = {}
            
            for color_name, (lower, upper) in self.color_ranges.items():
                lower_np = np.array(lower, dtype=np.uint8)
                upper_np = np.array(upper, dtype=np.uint8)
                
                mask = cv2.inRange(hsv, lower_np, upper_np)
                ratio = np.count_nonzero(mask) / (height * width)
                
                color_detections[color_name] = {
                    "ratio": float(ratio),
                    "present": ratio > 0.01  # At least 1% of image
                }
            
            # Calculate emergency color score
            emergency_color_score = (
                color_detections['red']['ratio'] * 3 +
                color_detections['orange']['ratio'] * 2 +
                color_detections['blue']['ratio'] * 1.5 +
                color_detections['yellow']['ratio'] * 1
            )
            
            color_detections['emergency_color_score'] = emergency_color_score
            
            return color_detections
            
        except Exception as e:
            print(f"Color analysis error: {e}")
            return {
                'red': {'ratio': 0, 'present': False},
                'orange': {'ratio': 0, 'present': False},
                'blue': {'ratio': 0, 'present': False},
                'yellow': {'ratio': 0, 'present': False},
                'emergency_color_score': 0
            }
    
    def analyze_scene_complexity(self, img) -> Dict[str, Any]:
        """Analyze scene complexity for incident assessment"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / (gray.size)
            
            # Texture analysis
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Brightness analysis
            brightness = np.mean(gray)
            brightness_category = "normal"
            if brightness < 50:
                brightness_category = "dark"
            elif brightness > 200:
                brightness_category = "bright"
            
            return {
                "edge_density": float(edge_density),
                "texture_variance": float(laplacian_var),
                "brightness": float(brightness),
                "brightness_category": brightness_category,
                "complexity_score": float(edge_density * 2 + laplacian_var / 100)
            }
        except Exception as e:
            print(f"Scene complexity error: {e}")
            return {
                "edge_density": 0.0,
                "texture_variance": 0.0,
                "brightness": 128.0,
                "brightness_category": "normal",
                "complexity_score": 0.0
            }
    
    def determine_severity(self, yolo_results: Dict, color_analysis: Dict) -> tuple:
        """Determine severity based on multiple factors"""
        base_score = 0
        
        # Object-based scoring
        base_score += yolo_results.get('emergency_score', 0) * 2
        
        # Color-based scoring
        color_score = color_analysis.get('emergency_color_score', 0)
        base_score += color_score * 3
        
        # Person count impact
        person_count = yolo_results.get('person_count', 0)
        if person_count > 5:
            base_score += 2
        elif person_count > 0:
            base_score += 1
        
        # Vehicle count impact
        vehicle_count = yolo_results.get('vehicle_count', 0)
        if vehicle_count > 3:
            base_score += 2
        elif vehicle_count > 0:
            base_score += 1
        
        # Normalize score
        severity_score = min(base_score / 10, 1.0)
        
        # Map to severity levels
        if severity_score > 0.7:
            return "Critical", severity_score
        elif severity_score > 0.5:
            return "High", severity_score
        elif severity_score > 0.3:
            return "Medium", severity_score
        else:
            return "Low", severity_score
    
    def default_response(self):
        return {
            "success": False,
            "severity": "Medium",
            "severity_score": 0.3,
            "confidence": 0.5,
            "detected_objects": [],
            "object_count": 0,
            "color_analysis": {},
            "scene_complexity": {},
            "error": "Image analysis failed"
        }
    
    def train_custom_model(self, dataset_path: str, epochs: int = 50):
        """Train custom YOLO model on incident dataset"""
        try:
            if not self.model:
                print("❌ YOLO not available for training")
                return False
            
            print(f"🚀 Training YOLO model on {dataset_path}")
            
            # Check if dataset exists
            if not os.path.exists(os.path.join(dataset_path, "data.yaml")):
                print("❌ Dataset config not found")
                return False
            
            # Train model
            self.model.train(
                data=os.path.join(dataset_path, "data.yaml"),
                epochs=epochs,
                imgsz=640,
                batch=16,
                name="incident_detection",
                project=self.model_path
            )
            
            print("✅ YOLO training complete!")
            return True
            
        except Exception as e:
            print(f"Training error: {e}")
            return False