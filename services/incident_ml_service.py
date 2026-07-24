import os
import uuid
from datetime import datetime
import json
from typing import Optional, Dict, Any
from fastapi import UploadFile

from ml_models.text_classifier import IncidentTextClassifier
from ml_models.image_processor import IncidentImageAnalyzer
from ml_models.video_analyzer import VideoAnalyzer

class IncidentMLService:
    def __init__(self):
        self.text_classifier = IncidentTextClassifier()
        self.image_analyzer = IncidentImageAnalyzer()
        self.video_analyzer = VideoAnalyzer()
        
        # Ensure upload directories exist
        os.makedirs("uploads/images", exist_ok=True)
        os.makedirs("uploads/videos", exist_ok=True)
        os.makedirs("uploads/temp", exist_ok=True)
    
    async def process_report(self, report_data: Dict, files: Optional[Dict] = None):
        """Process incident report with ML analysis"""
        report_id = str(uuid.uuid4())
        
        # Text analysis
        description = report_data.get("description", "")
        text_analysis = self.text_classifier.predict(description)
        keywords = self.text_classifier.extract_keywords(description)
        
        # Initialize media analysis
        image_analysis = None
        video_analysis = None
        media_paths = {}
        
        # Process uploaded files
        if files:
            for file_type, file in files.items():
                if file and file.filename:
                    try:
                        # Save file
                        file_path = await self.save_uploaded_file(file, report_id, file_type)
                        media_paths[file_type] = file_path
                        
                        # Analyze based on file type
                        if file_type == "image":
                            image_analysis = self.image_analyzer.analyze_image(file_path)
                        elif file_type == "video":
                            video_analysis = self.video_analyzer.analyze_video(file_path)
                    except Exception as e:
                        print(f"Error processing {file_type}: {e}")
        
        # Determine overall severity
        overall_severity = self.determine_overall_severity(
            text_analysis, image_analysis, video_analysis
        )
        
        # Calculate priority (1-5)
        priority = self.calculate_priority(
            overall_severity, 
            text_analysis.get("confidence", 0.5),
            report_data.get("barangay", "")
        )
        
        # Generate location info
        location_info = {
            "latitude": report_data.get("latitude", 0),
            "longitude": report_data.get("longitude", 0),
            "barangay": report_data.get("barangay", "Unknown"),
            "city": "Calapan",
            "province": "Oriental Mindoro",
            "country": "Philippines"
        }
        
        # Generate response
        response = {
            "success": True,
            "report_id": report_id,
            "incident_type": text_analysis.get("type", "Other"),
            "severity": overall_severity,
            "priority": priority,
            "confidence": text_analysis.get("confidence", 0.5),
            "analysis": {
                "text": text_analysis,
                "image": image_analysis,
                "video": video_analysis,
                "keywords": keywords
            },
            "location": location_info,
            "media_paths": media_paths,
            "timestamp": datetime.now().isoformat(),
            "message": "Incident report processed successfully"
        }
        
        # Add recommendations
        response["recommendations"] = self.generate_recommendations(
            text_analysis.get("type", "Other"),
            overall_severity,
            location_info.get("barangay")
        )
        
        return response
    
    async def save_uploaded_file(self, file: UploadFile, report_id: str, file_type: str):
        """Save uploaded file to disk"""
        # Generate safe filename
        safe_filename = "".join(c for c in file.filename if c.isalnum() or c in "._- ").rstrip()
        filename = f"{report_id}_{safe_filename}"
        
        # Determine upload directory
        if file_type == "image":
            upload_dir = "uploads/images"
        elif file_type == "video":
            upload_dir = "uploads/videos"
        else:
            upload_dir = "uploads/others"
        
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, filename)
        
        # Save file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        return file_path
    
    def determine_overall_severity(self, text_analysis, image_analysis, video_analysis):
        """Determine overall severity from all analyses"""
        severities = []
        weights = []
        
        # Text severity mapping with weight 1.0
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
        severities.append(text_severity)
        weights.append(1.0)
        
        # Image severity with weight 0.8
        if image_analysis and image_analysis.get("success"):
            severities.append(image_analysis.get("severity", "Medium"))
            weights.append(0.8)
        
        # Video severity with weight 0.7
        if video_analysis and video_analysis.get("success"):
            severities.append(video_analysis.get("overall_severity", "Medium"))
            weights.append(0.7)
        
        if not severities:
            return "Medium"
        
        # Weighted severity determination
        severity_scores = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
        weighted_sum = 0
        total_weight = sum(weights)
        
        for severity, weight in zip(severities, weights):
            weighted_sum += severity_scores.get(severity, 2) * weight
        
        avg_score = weighted_sum / total_weight if total_weight > 0 else 2
        
        # Map score back to severity
        if avg_score >= 3.5:
            return "Critical"
        elif avg_score >= 2.5:
            return "High"
        elif avg_score >= 1.5:
            return "Medium"
        else:
            return "Low"
    
    def calculate_priority(self, severity, confidence, barangay):
        """Calculate priority score (1-5)"""
        priority_map = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
        base_priority = priority_map.get(severity, 2)
        
        # Adjust based on confidence
        if confidence < 0.5:
            base_priority -= 0.5
        elif confidence > 0.8:
            base_priority += 0.5
        
        # Adjust based on barangay (prioritize certain areas)
        high_priority_barangays = ["Tawagan", "Sta. Isabel", "Lumangbayan", "Central", "Poblacion"]
        if barangay in high_priority_barangays:
            base_priority += 0.5
        
        # Clamp to 1-5 range
        return min(max(round(base_priority), 1), 5)
    
    def generate_recommendations(self, incident_type, severity, barangay):
        """Generate AI recommendations based on incident"""
        recommendations = []
        
        # General recommendations
        recommendations.append("🚨 Call 911 or local emergency number immediately")
        
        # Type-specific recommendations
        if incident_type == "Accident":
            recommendations.extend([
                "🚗 Secure the accident scene to prevent further collisions",
                "🏥 Check for injuries and provide first aid if trained",
                "📱 Take photos of the scene for documentation",
                "🚓 Wait for police and ambulance to arrive"
            ])
        elif incident_type == "Fire":
            recommendations.extend([
                "🔥 Evacuate the area immediately",
                "📞 Call fire department: 288-3333",
                "🚒 Do not attempt to fight large fires yourself",
                "🌬️ Stay upwind of smoke and fumes"
            ])
        elif incident_type == "Medical":
            recommendations.extend([
                "🏥 Call ambulance: 288-1111",
                "💊 Do not move the patient unless in immediate danger",
                "📱 Keep emergency contacts ready",
                "🩺 Provide clear location and symptoms to dispatcher"
            ])
        elif incident_type == "Crime":
            recommendations.extend([
                "🚓 Call police: 288-4444",
                "📱 Do not confront suspects",
                "👀 Note suspect descriptions and direction of escape",
                "📸 Take photos/videos if safe to do so"
            ])
        
        # Severity-based recommendations
        if severity in ["High", "Critical"]:
            recommendations.extend([
                "⚠️ Expect emergency response within 5-10 minutes",
                "📢 Alert neighbors if safe to do so",
                "🆘 Use RESQAPP's live chat for real-time updates"
            ])
        
        # Barangay-specific recommendations
        if barangay == "Tawagan":
            recommendations.append("📍 Nearest hospital: Oriental Mindoro Provincial Hospital (2km)")
        elif barangay == "Sta. Isabel":
            recommendations.append("📍 Nearest fire station: Calapan City Fire Station (1.5km)")
        
        return recommendations
    
    async def analyze_text_only(self, text: str):
        """Analyze text without creating a report"""
        if len(text.strip()) < 3:
            return {
                "type": "Other",
                "confidence": 0.5,
                "keywords": [],
                "all_predictions": [0.14] * 7
            }
        
        analysis = self.text_classifier.predict(text)
        keywords = self.text_classifier.extract_keywords(text)
        
        return {
            "type": analysis.get("type", "Other"),
            "confidence": analysis.get("confidence", 0.5),
            "keywords": keywords,
            "all_predictions": analysis.get("all_predictions", []),
            "color": self.text_classifier.get_confidence_color(analysis.get("confidence", 0.5))
        }