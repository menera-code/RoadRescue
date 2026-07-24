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

# Use predictor.py instead of ml_models
from services.predictor import predict_text, analyze_image, analyze_video

class EnhancedIncidentMLService:
    def __init__(self, use_enhanced: bool = True):
        self.use_enhanced = use_enhanced
        print("🚀 Using predictor.py for ML services")

        # Ensure upload directories exist
        os.makedirs("uploads/images", exist_ok=True)
        os.makedirs("uploads/videos", exist_ok=True)
        os.makedirs("uploads/temp", exist_ok=True)

        # Training data collector
        self.training_buffer = []
        self.buffer_limit = 100

    async def enhanced_text_analysis(self, text: str) -> Dict[str, Any]:
        """Analyze text using predictor.py"""
        if len(text.strip()) < 3:
            return self.default_text_analysis()
        
        # Use predictor.py
        result = predict_text(text)
        
        return {
            "type": result.get("incident_type", "Accident"),
            "confidence": result.get("type_confidence", 1.0),
            "severity": result.get("severity", "medium"),
            "severity_confidence": result.get("severity_confidence", 0.5),
            "keywords": result.get("mentioned_vehicles", []),
            "all_predictions": result.get("all_severity_scores", {}),
            "word_count": len(text.split()),
            "has_urgency": any(w in text.lower() for w in ["urgent", "emergency", "help", "immediately", "now"]),
            "model": "predictor",
            "success": True
        }

    async def analyze_image_with_model(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Analyze image using predictor.py YOLO"""
        try:
            return analyze_image(image_path)
        except Exception as e:
            print(f"Image analysis error: {e}")
            return None

    async def analyze_video_with_image_model(self, video_path: str, num_frames=8) -> Dict[str, Any]:
        """Analyze video using predictor.py YOLO"""
        try:
            result = analyze_video(video_path, frame_interval=30)
            return {
                "success": True,
                "summary": result.get("summary", []),
                "total_frames_analyzed": result.get("total_frames_analyzed", 0),
                "vehicles": result.get("vehicles", {})
            }
        except Exception as e:
            print(f"Video analysis error: {e}")
            return {"success": False, "error": str(e)}

    async def process_report(self, report_data: Dict, files: Optional[Dict] = None):
        """Process incident report with ML analysis"""
        report_id = str(uuid.uuid4())
        
        # Text analysis
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
                            image_analysis = await self.analyze_image_with_model(file_path)
                        elif file_type == "video":
                            video_analysis = await self.analyze_video_with_image_model(file_path)
                    except Exception as e:
                        print(f"Error processing {file_type}: {e}")
        
        # Determine severity
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
            "incident_type": text_analysis.get("type", "Accident"),
            "severity": overall_severity,
            "priority": priority,
            "confidence": text_analysis.get("confidence", 0.5),
            "model_type": "predictor",
            "analysis": {
                "text": text_analysis,
                "image": image_analysis,
                "video": video_analysis,
                "text_keywords": text_analysis.get("keywords", []),
                "image_objects": image_analysis.get("objects", []) if image_analysis else [],
                "video_vehicles": video_analysis.get("vehicles", {}) if video_analysis else {}
            },
            "location": location_info,
            "media_paths": media_paths,
            "timestamp": datetime.now().isoformat(),
            "message": "Incident report processed successfully"
        }
        
        # Add recommendations
        response["recommendations"] = self.enhanced_generate_recommendations(
            text_analysis.get("type", "Accident"),
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

    # ==================== HELPER METHODS ====================

    def enhanced_determine_severity(self, text_analysis, image_analysis, 
                                   video_analysis, report_data) -> str:
        """Enhanced severity determination"""
        scores = []
        weights = []

        # Text severity
        text_severity = text_analysis.get("severity", "medium").capitalize()
        text_confidence = text_analysis.get("severity_confidence", 0.5)
        text_weight = 0.8 + (text_confidence * 0.4)
        scores.append(self.severity_to_score(text_severity))
        weights.append(text_weight)

        # Image severity
        if image_analysis and image_analysis.get("objects"):
            vehicle_count = len(image_analysis.get("vehicles", {}))
            if vehicle_count > 3:
                img_severity = "High"
                img_weight = 0.8
            elif vehicle_count > 1:
                img_severity = "Medium"
                img_weight = 0.6
            else:
                img_severity = "Low"
                img_weight = 0.4
            scores.append(self.severity_to_score(img_severity))
            weights.append(img_weight)

        # Video severity
        if video_analysis and video_analysis.get("success"):
            vehicle_count = sum(video_analysis.get("vehicles", {}).values())
            if vehicle_count > 5:
                vid_severity = "High"
            elif vehicle_count > 2:
                vid_severity = "Medium"
            else:
                vid_severity = "Low"
            scores.append(self.severity_to_score(vid_severity))
            weights.append(0.7)

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

        if image_analysis and image_analysis.get("objects"):
            vehicle_count = len(image_analysis.get("vehicles", {}))
            if vehicle_count > 3:
                base_priority += 0.5

        if video_analysis and video_analysis.get("success"):
            vehicle_count = sum(video_analysis.get("vehicles", {}).values())
            if vehicle_count > 5:
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
        return location

    def enhanced_generate_recommendations(self, incident_type, severity, 
                                        location, image_analysis, video_analysis) -> List[str]:
        recommendations = ["🚨 Call 911 or local emergency number immediately"]

        if incident_type == "Accident":
            recommendations.extend([
                "🚗 Secure accident scene with hazard lights or warning signs",
                "🏥 Check for injuries - do not move seriously injured persons",
                "📱 Document scene with photos from multiple angles",
                "🚓 Await police for official report"
            ])

        if severity in ["Critical", "High"]:
            recommendations.extend([
                "⚠️ Multiple emergency units dispatched",
                "🕐 Expected response time: 5-10 minutes",
                "📢 Use RESQAPP live chat for real-time coordination"
            ])

        barangay = location.get("barangay", "")
        if barangay == "Tawagan":
            recommendations.extend([
                "📍 Nearest hospital: Oriental Mindoro Provincial Hospital (2.1km)",
                "🚑 Ambulance contact: (043) 288-2000"
            ])

        hour = datetime.now().hour
        if 22 <= hour <= 5:
            recommendations.append("🌙 Nighttime incident - use flashlight for visibility")

        return recommendations[:8]

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

    def calculate_risk_assessment(self, severity, priority, barangay, analysis) -> Dict[str, Any]:
        risk_score = priority * 2
        risk_score = min(max(risk_score, 1), 10)
        return {
            "score": risk_score,
            "level": "low" if risk_score < 4 else "medium" if risk_score < 7 else "high",
            "factors": ["severity", "priority", "location"],
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
            "severity": text_analysis.get("severity"),
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

    def default_text_analysis(self):
        return {
            "type": "Accident",
            "confidence": 1.0,
            "severity": "medium",
            "severity_confidence": 0.5,
            "keywords": [],
            "all_predictions": {},
            "word_count": 0,
            "has_urgency": False,
            "model": "predictor",
            "success": True
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
        """Analyze text without creating a report"""
        return await self.enhanced_text_analysis(text)