import os
import uuid
from datetime import datetime
import json
from typing import Optional, Dict, Any, List
from fastapi import UploadFile
import asyncio
import cv2
import numpy as np
from PIL import Image

# Import enhanced models
try:
    from ml_models.text_classifier_enhanced import EnhancedIncidentClassifier
    from ml_models.image_analyzer_enhanced import EnhancedImageAnalyzer
    from ml_models.video_analyzer_enhanced import EnhancedVideoAnalyzer
    ENHANCED_MODELS_AVAILABLE = True
except ImportError:
    # Fallback to original models
    from ml_models.text_classifier import IncidentTextClassifier
    from ml_models.image_processor import IncidentImageAnalyzer
    from ml_models.video_analyzer import VideoAnalyzer
    ENHANCED_MODELS_AVAILABLE = False
    print("⚠ Enhanced models not available, using basic models")

# Try to import deep learning libraries
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import joblib
    from torchvision import transforms, models as torch_models
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    print("⚠ Deep learning libraries not installed, skipping BERT/EfficientNet")

class EnhancedIncidentMLService:
    def __init__(self, use_enhanced: bool = True):
        self.use_enhanced = use_enhanced and ENHANCED_MODELS_AVAILABLE
        self.deep_learning_available = DEEP_LEARNING_AVAILABLE

        # Initialize existing ML components
        if self.use_enhanced:
            print("🚀 Using enhanced ML models")
            self.text_classifier = EnhancedIncidentClassifier()
            self.image_analyzer = EnhancedImageAnalyzer()
            self.video_analyzer = EnhancedVideoAnalyzer(self.image_analyzer)
        else:
            print("⚠ Using basic ML models")
            self.text_classifier = IncidentTextClassifier()
            self.image_analyzer = IncidentImageAnalyzer()
            self.video_analyzer = VideoAnalyzer()

        # Initialize deep learning models
        self.text_model = None
        self.text_tokenizer = None
        self.text_label_encoder = None
        self.image_model = None
        self.image_label_encoder = None
        self.image_transform = None
        self.device = None

        if self.deep_learning_available:
            self._load_deep_learning_models()

        # Ensure upload directories exist
        os.makedirs("uploads/images", exist_ok=True)
        os.makedirs("uploads/videos", exist_ok=True)
        os.makedirs("uploads/temp", exist_ok=True)

        # Training data collector
        self.training_buffer = []
        self.buffer_limit = 100

    def _load_deep_learning_models(self):
        """Load BERT and EfficientNet models if they exist."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Using device: {self.device}")

        # Load text model (BERT)
        text_model_path = './models/text/bert_final'
        if os.path.exists(text_model_path):
            try:
                self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_path)
                self.text_model = AutoModelForSequenceClassification.from_pretrained(text_model_path)
                self.text_model.to(self.device)
                self.text_model.eval()
                self.text_label_encoder = joblib.load('./models/text/label_encoder.pkl')
                print("✅ BERT text model loaded")
            except Exception as e:
                print(f"❌ Failed to load BERT model: {e}")

        # Load image model (EfficientNet)
        image_model_path = './models/image/efficientnet_final.pth'
        label_encoder_path = './models/image/label_encoder.pkl'
        if os.path.exists(image_model_path) and os.path.exists(label_encoder_path):
            try:
                self.image_label_encoder = joblib.load(label_encoder_path)
                num_classes = len(self.image_label_encoder.classes_)
                self.image_model = torch_models.efficientnet_b0(pretrained=False)
                num_ftrs = self.image_model.classifier[1].in_features
                self.image_model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
                self.image_model.load_state_dict(torch.load(image_model_path, map_location=self.device))
                self.image_model.to(self.device)
                self.image_model.eval()
                self.image_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                print("✅ EfficientNet image model loaded")
            except Exception as e:
                print(f"❌ Failed to load image model: {e}")

    async def enhanced_text_analysis(self, text: str) -> Dict[str, Any]:
        """Enhanced text analysis using BERT if available, otherwise fallback."""
        if len(text.strip()) < 3:
            return self.default_text_analysis()

        # Try to use BERT
        if self.text_model and self.text_tokenizer:
            try:
                inputs = self.text_tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.text_model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                predicted_idx = probs.argmax()
                confidence = probs[predicted_idx]
                predicted_type = self.text_label_encoder.inverse_transform([predicted_idx])[0]

                all_predictions = {
                    self.text_label_encoder.classes_[i]: float(probs[i])
                    for i in range(len(probs))
                }

                # Extract keywords (optional)
                keywords = []
                if hasattr(self.text_classifier, 'extract_keywords'):
                    keywords = self.text_classifier.extract_keywords(text)
                elif hasattr(self.text_classifier, 'rule_based') and hasattr(self.text_classifier.rule_based, 'extract_keywords'):
                    keywords = self.text_classifier.rule_based.extract_keywords(text)

                return {
                    "type": predicted_type,
                    "confidence": float(confidence),
                    "keywords": keywords,
                    "all_predictions": all_predictions,
                    "model": "bert",
                    "success": True,
                    "word_count": len(text.split()),
                    "has_urgency": any(w in text.lower() for w in ["urgent", "emergency", "help", "immediately", "now"])
                }
            except Exception as e:
                print(f"BERT inference error: {e}")
                # Fall through to existing classifier

        # Fallback to existing text classifier
        analysis = self.text_classifier.predict(text)

        # Add keyword extraction if missing
        if "keywords" not in analysis:
            if hasattr(self.text_classifier, 'extract_keywords'):
                analysis["keywords"] = self.text_classifier.extract_keywords(text)
            elif hasattr(self.text_classifier, 'rule_based') and hasattr(self.text_classifier.rule_based, 'extract_keywords'):
                analysis["keywords"] = self.text_classifier.rule_based.extract_keywords(text)
            else:
                analysis["keywords"] = []

        analysis["word_count"] = len(text.split())
        analysis["has_urgency"] = any(w in text.lower() for w in ["urgent", "emergency", "help", "immediately", "now"])
        return analysis

    async def analyze_image_with_model(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Use image model if available, else None."""
        if not self.image_model or not self.image_transform:
            return None
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.image_transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.image_model(img_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_idx = probs.argmax()
            confidence = probs[predicted_idx]
            predicted_type = self.image_label_encoder.inverse_transform([predicted_idx])[0]
            return {
                "type": predicted_type,
                "confidence": float(confidence),
                "all_predictions": {
                    self.image_label_encoder.classes_[i]: float(probs[i])
                    for i in range(len(probs))
                },
                "model": "efficientnet",
                "success": True,
                "file_path": image_path
            }
        except Exception as e:
            print(f"Image analysis error: {e}")
            return None

    async def analyze_video_with_image_model(self, video_path: str, num_frames=8) -> Dict[str, Any]:
        """Extract frames, classify each, and aggregate."""
        if not self.image_model or not self.image_transform:
            # Fallback to existing video analyzer
            return self.video_analyzer.analyze_video(video_path)

        # Extract frames
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"success": False, "error": "Could not open video"}
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return {"success": False, "error": "Video has no frames"}
        step = max(1, total_frames // num_frames)
        frames = []
        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb).resize((224, 224), Image.LANCZOS)
                frames.append(pil_img)
            if len(frames) >= num_frames:
                break
        cap.release()
        if not frames:
            return {"success": False, "error": "Could not extract frames"}

        # Classify each frame
        frame_probs = []
        for frame in frames:
            img_tensor = self.image_transform(frame).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.image_model(img_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            frame_probs.append(probs)

        # Aggregate
        avg_probs = np.mean(frame_probs, axis=0)
        predicted_idx = np.argmax(avg_probs)
        confidence = avg_probs[predicted_idx]
        predicted_type = self.image_label_encoder.inverse_transform([predicted_idx])[0]

        # Progression analysis
        half = len(frame_probs) // 2
        if half > 0:
            first_half = np.mean(frame_probs[:half], axis=0)[predicted_idx]
            second_half = np.mean(frame_probs[half:], axis=0)[predicted_idx]
            if second_half > first_half * 1.2:
                progression = "escalating"
            elif first_half > second_half * 1.2:
                progression = "de-escalating"
            else:
                progression = "stable"
        else:
            progression = "unknown"

        all_predictions = {
            self.image_label_encoder.classes_[i]: float(avg_probs[i])
            for i in range(len(avg_probs))
        }

        return {
            "success": True,
            "type": predicted_type,
            "confidence": float(confidence),
            "all_predictions": all_predictions,
            "frames_analyzed": len(frames),
            "incident_progression": progression,
            "temporal_patterns": {
                "first_half": float(first_half) if half > 0 else None,
                "second_half": float(second_half) if half > 0 else None
            }
        }

    async def process_report(self, report_data: Dict, files: Optional[Dict] = None):
        """Process incident report with enhanced ML analysis (including deep learning)."""
        report_id = str(uuid.uuid4())

        # Enhanced text analysis (uses BERT if available)
        description = report_data.get("description", "")
        text_analysis = await self.enhanced_text_analysis(description)

        # Initialize media analysis
        image_analysis = None
        video_analysis = None
        media_paths = {}

        # Process uploaded files
        if files:
            for file_type, file in files.items():
                if file and file.filename:
                    try:
                        file_path = await self.save_uploaded_file(file, report_id, file_type)
                        media_paths[file_type] = file_path

                        if file_type == "image":
                            # Try deep learning model first
                            img_analysis = await self.analyze_image_with_model(file_path)
                            if img_analysis:
                                image_analysis = img_analysis
                            else:
                                image_analysis = self.image_analyzer.analyze_image(file_path)
                        elif file_type == "video":
                            video_analysis = await self.analyze_video_with_image_model(file_path)
                    except Exception as e:
                        print(f"Error processing {file_type}: {e}")

        # Enhanced severity determination (unchanged from original)
        overall_severity = self.enhanced_determine_severity(
            text_analysis, image_analysis, video_analysis, report_data
        )
        priority = self.enhanced_calculate_priority(
            overall_severity,
            text_analysis.get("confidence", 0.5),
            report_data.get("barangay", ""),
            image_analysis,
            video_analysis
        )
        location_info = self.enhanced_location_analysis(report_data)

        # Build response
        response = {
            "success": True,
            "report_id": report_id,
            "incident_type": text_analysis.get("type", "Other"),
            "severity": overall_severity,
            "priority": priority,
            "confidence": text_analysis.get("confidence", 0.5),
            "model_type": "deep_learning" if (self.text_model or self.image_model) else "enhanced" if self.use_enhanced else "basic",
            "analysis": {
                "text": text_analysis,
                "image": image_analysis,
                "video": video_analysis,
                "text_keywords": text_analysis.get("keywords", []),
                "image_objects": image_analysis.get("detected_objects", []) if image_analysis else [],
                "video_patterns": video_analysis.get("temporal_patterns", {}) if video_analysis else {}
            },
            "location": location_info,
            "media_paths": media_paths,
            "timestamp": datetime.now().isoformat(),
            "message": "Incident report processed with enhanced ML analysis"
        }

        # Enhanced AI recommendations (unchanged)
        response["recommendations"] = self.enhanced_generate_recommendations(
            text_analysis.get("type", "Other"),
            overall_severity,
            location_info,
            image_analysis,
            video_analysis
        )
        response["risk_assessment"] = self.calculate_risk_assessment(
            overall_severity,
            priority,
            location_info.get("barangay"),
            response["analysis"]
        )

        # Collect for training
        await self.collect_training_data(
            report_id, description, text_analysis, image_analysis, video_analysis
        )

        return response

    # ==================== EXISTING METHODS (UNCHANGED) ====================

    def enhanced_determine_severity(self, text_analysis, image_analysis, 
                                   video_analysis, report_data) -> str:
        """Enhanced severity determination with multi-modal fusion"""
        scores = []
        weights = []

        # Text severity with adaptive weight
        type_to_severity = {
            "Accident": "Medium",
            "Fire": "High",
            "Medical": "High",
            "Crime": "Medium",
            "Natural Disaster": "Critical",
            "Infrastructure": "Low",
            "Other": "Low"
        }

        text_severity = type_to_severity.get(text_analysis.get("type", "Other"), "Medium")
        text_confidence = text_analysis.get("confidence", 0.5)

        text_weight = 0.8 + (text_confidence * 0.4)  # 0.8-1.2 range
        scores.append(self.severity_to_score(text_severity))
        weights.append(text_weight)

        # Image severity with enhanced scoring
        if image_analysis and image_analysis.get("success"):
            img_severity = image_analysis.get("severity", "Medium")
            img_confidence = image_analysis.get("confidence", 0.5)

            if "severity_score" in image_analysis:
                img_score = image_analysis["severity_score"]
                img_weight = 0.6 + (img_confidence * 0.4) + (img_score * 0.2)
            else:
                img_weight = 0.6 + (img_confidence * 0.4)

            scores.append(self.severity_to_score(img_severity))
            weights.append(img_weight)

        # Video severity with temporal consideration
        if video_analysis and video_analysis.get("success"):
            vid_severity = video_analysis.get("overall_severity", "Medium")
            vid_progression = video_analysis.get("incident_progression", {})

            progression_weight = 1.0
            if vid_progression == "escalating":
                progression_weight = 1.2

            vid_weight = 0.7 * progression_weight
            scores.append(self.severity_to_score(vid_severity))
            weights.append(vid_weight)

        # Location-based adjustment
        barangay = report_data.get("barangay", "")
        if barangay in ["Tawagan", "Sta. Isabel", "Poblacion"]:
            scores.append(self.severity_to_score("Medium"))
            weights.append(0.3)

        if not scores:
            return "Medium"

        weighted_avg = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        return self.score_to_severity(weighted_avg)

    def enhanced_calculate_priority(self, severity, text_confidence, barangay,
                                   image_analysis, video_analysis) -> int:
        severity_map = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
        base_priority = severity_map.get(severity, 2)

        if text_confidence < 0.4:
            base_priority -= 1
        elif text_confidence > 0.8:
            base_priority += 0.5

        if image_analysis and image_analysis.get("success"):
            img_objects = len(image_analysis.get("detected_objects", []))
            if img_objects > 5:
                base_priority += 0.5
            if image_analysis.get("severity") == "Critical":
                base_priority += 1

        if video_analysis and video_analysis.get("success"):
            if video_analysis.get("incident_progression") == "escalating":
                base_priority += 0.5

        high_priority_barangays = ["Tawagan", "Sta. Isabel", "Lumangbayan", "Central", "Poblacion"]
        if barangay in high_priority_barangays:
            base_priority += 0.5

        hour = datetime.now().hour
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            base_priority += 0.5
        elif 22 <= hour <= 5:
            base_priority += 0.3

        return min(max(round(base_priority), 1), 5)

    def enhanced_location_analysis(self, report_data: Dict) -> Dict[str, Any]:
        location = {
            "latitude": report_data.get("latitude", 0),
            "longitude": report_data.get("longitude", 0),
            "barangay": report_data.get("barangay", "Unknown"),
            "city": "Calapan",
            "province": "Oriental Mindoro",
            "country": "Philippines",
            "coordinates_valid": self.validate_coordinates(
                report_data.get("latitude", 0),
                report_data.get("longitude", 0)
            )
        }

        location["proximity_to_hospital"] = self.calculate_proximity(
            location["latitude"], location["longitude"],
            13.414, 121.180  # Example hospital coordinates
        )
        location["proximity_to_fire_station"] = self.calculate_proximity(
            location["latitude"], location["longitude"],
            13.412, 121.182  # Example fire station coordinates
        )
        return location

    def enhanced_generate_recommendations(self, incident_type, severity, 
                                        location, image_analysis, video_analysis) -> List[str]:
        recommendations = []
        recommendations.append("🚨 Call 911 or local emergency number immediately")

        if incident_type == "Accident":
            recommendations.extend([
                "🚗 Secure accident scene with hazard lights or warning signs",
                "🏥 Check for injuries - do not move seriously injured persons",
                "📱 Document scene with photos from multiple angles",
                "🚓 Await police for official report",
                "🩺 Provide first aid if trained - focus on ABC (Airway, Breathing, Circulation)"
            ])
            if image_analysis and "vehicle_count" in image_analysis:
                if image_analysis["vehicle_count"] > 2:
                    recommendations.append("⚠️ Multiple vehicles involved - expect traffic congestion")

        elif incident_type == "Fire":
            recommendations.extend([
                "🔥 Evacuate immediately - do not retrieve possessions",
                "📞 Call fire department: 288-3333",
                "🚒 Close doors behind you to slow fire spread",
                "🌬️ Stay low to avoid smoke inhalation",
                "🔥 If trapped, signal from window with light-colored cloth"
            ])
            if severity in ["High", "Critical"]:
                recommendations.append("🆘 Alert neighbors to evacuate if safe to do so")

        if severity == "Critical":
            recommendations.extend([
                "⚠️ CRITICAL: Multiple emergency units dispatched",
                "🕐 Expected response time: 3-5 minutes",
                "📢 Use RESQAPP live chat for real-time coordination",
                "🚁 Consider air ambulance if remote location"
            ])

        barangay = location.get("barangay", "")
        if barangay == "Tawagan":
            recommendations.extend([
                "📍 Nearest hospital: Oriental Mindoro Provincial Hospital (2.1km)",
                "🚑 Ambulance contact: (043) 288-2000",
                "🛣️ Access via National Road - clear route available"
            ])

        hour = datetime.now().hour
        if 22 <= hour <= 5:
            recommendations.append("🌙 Nighttime incident - use flashlight for visibility")

        if image_analysis and image_analysis.get("object_count", 0) > 10:
            recommendations.append("📸 Complex scene detected - additional photos helpful")

        if video_analysis and video_analysis.get("incident_progression") == "escalating":
            recommendations.append("📈 Situation escalating - prepare for worsening conditions")

        return recommendations[:10]

    def severity_to_score(self, severity: str) -> float:
        scores = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
        return scores.get(severity, 2)

    def score_to_severity(self, score: float) -> str:
        if score >= 3.5:
            return "Critical"
        elif score >= 2.5:
            return "High"
        elif score >= 1.5:
            return "Medium"
        else:
            return "Low"

    def validate_coordinates(self, lat: float, lng: float) -> bool:
        return -90 <= lat <= 90 and -180 <= lng <= 180

    def calculate_proximity(self, lat1: float, lng1: float, lat2: float, lng2: float) -> str:
        distance = ((lat1 - lat2) ** 2 + (lng1 - lng2) ** 2) ** 0.5 * 111  # Approx km
        if distance < 1:
            return "very_close"
        elif distance < 3:
            return "close"
        elif distance < 5:
            return "moderate"
        else:
            return "far"

    def calculate_risk_assessment(self, severity, priority, barangay, analysis) -> Dict[str, Any]:
        risk_score = priority * 2
        text_conf = analysis.get("text", {}).get("confidence", 0.5)
        risk_score *= (0.5 + text_conf)
        high_risk_barangays = ["Tawagan", "Sta. Isabel"]
        if barangay in high_risk_barangays:
            risk_score *= 1.2
        risk_score = min(max(risk_score, 1), 10)

        return {
            "score": risk_score,
            "level": "low" if risk_score < 4 else "medium" if risk_score < 7 else "high",
            "factors": ["severity", "priority", "location", "confidence"],
            "interpretation": self.interpret_risk_score(risk_score)
        }

    def interpret_risk_score(self, score: float) -> str:
        if score >= 8:
            return "Extreme risk - Immediate intervention required"
        elif score >= 6:
            return "High risk - Urgent response needed"
        elif score >= 4:
            return "Moderate risk - Timely response required"
        else:
            return "Low risk - Standard response appropriate"

    async def collect_training_data(self, report_id, description, text_analysis, 
                                   image_analysis, video_analysis):
        training_item = {
            "report_id": report_id,
            "description": description,
            "incident_type": text_analysis.get("type"),
            "text_confidence": text_analysis.get("confidence"),
            "timestamp": datetime.now().isoformat()
        }
        self.training_buffer.append(training_item)
        if len(self.training_buffer) >= self.buffer_limit:
            await self.save_training_data()

    async def save_training_data(self):
        if not self.training_buffer:
            return
        try:
            filename = f"training_data/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs("training_data", exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(self.training_buffer, f, indent=2)
            print(f"💾 Saved {len(self.training_buffer)} training samples to {filename}")
            self.training_buffer = []
        except Exception as e:
            print(f"Error saving training data: {e}")

    def get_user_ml_stats(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        from backend.ml_analytics import ml_analytics
        from database import SessionLocal
        db = SessionLocal()
        try:
            return ml_analytics.get_user_ml_stats(user_id, days, db)
        finally:
            db.close()

    async def train_models(self, training_data_path: str = None):
        if not self.use_enhanced:
            print("⚠ Cannot train basic models")
            return
        print("🚀 Starting model training...")
        if hasattr(self.text_classifier, 'train_bert'):
            print("📝 Training BERT model...")
            synthetic_data = self.text_classifier.generate_synthetic_data()
            results = self.text_classifier.train_bert(synthetic_data)
            print(f"✅ BERT training results: {results}")
        if hasattr(self.image_analyzer, 'train_custom_model') and training_data_path:
            print("🖼️ Training image model...")
            self.image_analyzer.train_custom_model(training_data_path)
        print("🎉 Model training complete!")

    def default_text_analysis(self):
        return {
            "type": "Other",
            "confidence": 0.5,
            "keywords": [],
            "all_predictions": [0.14] * 7,
            "word_count": 0,
            "has_urgency": False
        }

    async def save_uploaded_file(self, file: UploadFile, report_id: str, file_type: str):
        safe_filename = "".join(c for c in file.filename if c.isalnum() or c in "._- ").rstrip()
        filename = f"{report_id}_{safe_filename}"
        if file_type == "image":
            upload_dir = "uploads/images"
        elif file_type == "video":
            upload_dir = "uploads/videos"
        else:
            upload_dir = "uploads/others"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        return file_path

    async def analyze_text_only(self, text: str):
        """Analyze text without creating a report (uses BERT if available)"""
        return await self.enhanced_text_analysis(text)