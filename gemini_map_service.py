# gemini_map_service.py
import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import google.genai as genai
import requests
import random

load_dotenv()

class GeminiMapService:
    """AI-powered map service with Gemini integration"""
    
    def __init__(self):
        # Initialize Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
            self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            print(f"✅ Gemini Map Service initialized with model: {self.model}")
        else:
            self.client = None
            print("⚠️ GEMINI_API_KEY not found, using simulated AI features")
        
        # External APIs (you can add real APIs later)
        self.weather_api_key = os.getenv("WEATHER_API_KEY")
        self.traffic_api_key = os.getenv("TRAFFIC_API_KEY", "demo")
        
        # Cache for frequent queries
        self.route_cache = {}
        self.insights_cache = {}
        
    async def get_ai_route_analysis(
        self,
        start_lat: float,
        start_lng: float,
        dest_lat: float,
        dest_lng: float,
        optimize_for: str = "fastest",
        user_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Get AI-powered route analysis with real-time insights
        """
        try:
            # Get base route from OSRM
            base_route = await self.get_osrm_route(start_lat, start_lng, dest_lat, dest_lng)
            
            # Get real-time data
            realtime_data = await self.get_realtime_data(
                start_lat, start_lng, dest_lat, dest_lng
            )
            
            # Generate AI insights
            ai_insights = await self.generate_ai_insights(
                base_route, realtime_data, optimize_for, user_context
            )
            
            # Find alternative routes with AI analysis
            alternatives = await self.get_ai_alternatives(
                start_lat, start_lng, dest_lat, dest_lng,
                base_route, realtime_data, optimize_for
            )
            
            # Predict traffic conditions
            traffic_prediction = await self.predict_traffic_with_ai(
                base_route, realtime_data
            )
            
            # Generate personalized recommendations
            recommendations = await self.generate_recommendations(
                base_route, realtime_data, ai_insights, user_context
            )
            
            return {
                "success": True,
                "primary_route": {
                    **base_route,
                    "ai_score": self.calculate_ai_score(base_route, optimize_for),
                    "confidence": random.uniform(0.7, 0.95)
                },
                "traffic_prediction": traffic_prediction,
                "ai_insights": ai_insights,
                "alternative_routes": alternatives,
                "recommendations": recommendations,
                "realtime_data": realtime_data,
                "optimization_type": optimize_for,
                "estimated_time_savings": self.calculate_time_savings(base_route, alternatives),
                "timestamp": datetime.now().isoformat(),
                "ai_model_used": self.model if self.client else "simulated"
            }
            
        except Exception as e:
            print(f"❌ AI route analysis error: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_realtime_data(
        self,
        start_lat: float,
        start_lng: float,
        dest_lat: float,
        dest_lng: float
    ) -> Dict[str, Any]:
        """Get real-time traffic, weather, and incident data"""
        try:
            # Simulated real-time data (replace with actual APIs)
            realtime_data = {
                "traffic_level": random.choice(["low", "medium", "high"]),
                "average_speed": random.randint(20, 80),
                "incidents": await self.get_simulated_incidents(start_lat, start_lng, dest_lat, dest_lng),
                "weather": await self.get_simulated_weather(start_lat, start_lng),
                "road_conditions": self.get_simulated_road_conditions(),
                "time_of_day": datetime.now().hour,
                "day_of_week": datetime.now().strftime("%A"),
                "last_updated": datetime.now().isoformat()
            }
            
            # Add AI analysis if Gemini is available
            if self.client:
                ai_traffic_analysis = await self.analyze_traffic_patterns(
                    start_lat, start_lng, dest_lat, dest_lng, realtime_data
                )
                realtime_data["ai_analysis"] = ai_traffic_analysis
            
            return realtime_data
            
        except Exception as e:
            print(f"Realtime data error: {e}")
            return self.get_fallback_realtime_data()
    
    async def generate_ai_insights(
        self,
        route_data: Dict,
        realtime_data: Dict,
        optimize_for: str,
        user_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Generate AI-powered insights about the route"""
        
        if not self.client:
            return self.get_simulated_insights(route_data, realtime_data, optimize_for)
        
        try:
            # Prepare prompt for Gemini
            prompt = f"""
            Analyze this route for Calapan City emergency response:
            
            ROUTE INFORMATION:
            - Distance: {route_data.get('distance', 0) / 1000:.1f} km
            - Estimated Time: {route_data.get('duration', 0) / 60:.0f} minutes
            - Optimization Goal: {optimize_for}
            
            REAL-TIME CONDITIONS:
            - Traffic Level: {realtime_data.get('traffic_level', 'unknown')}
            - Time of Day: {realtime_data.get('time_of_day', 12)}:00
            - Day of Week: {realtime_data.get('day_of_week', 'Monday')}
            - Weather: {realtime_data.get('weather', {}).get('condition', 'Clear')}
            - Incidents: {len(realtime_data.get('incidents', []))} reported
            
            USER CONTEXT:
            {json.dumps(user_context or {}, indent=2)}
            
            Please provide:
            1. Safety assessment (1-10 score)
            2. Emergency vehicle accessibility
            3. Potential delays and why
            4. Alternative suggestions
            5. Weather impact analysis
            6. Time-sensitive recommendations
            
            Format as JSON with these keys: safety_score, emergency_access, potential_delays, 
            alternatives, weather_impact, recommendations, confidence
            """
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=1000,
                    response_mime_type="application/json"
                )
            )
            
            # Parse JSON response
            insights = json.loads(response.text)
            return insights
            
        except Exception as e:
            print(f"AI insights error: {e}")
            return self.get_simulated_insights(route_data, realtime_data, optimize_for)
    
    async def predict_traffic_with_ai(
        self,
        route_data: Dict,
        realtime_data: Dict
    ) -> Dict[str, Any]:
        """Predict future traffic conditions using AI"""
        
        if not self.client:
            return self.get_simulated_traffic_prediction(route_data)
        
        try:
            prompt = f"""
            Predict traffic conditions for this route in Calapan City:
            
            Current Conditions:
            - Traffic: {realtime_data.get('traffic_level', 'medium')}
            - Time: {datetime.now().strftime('%I:%M %p')}
            - Day: {datetime.now().strftime('%A')}
            - Route Distance: {route_data.get('distance', 0) / 1000:.1f} km
            
            Historical Patterns:
            - Morning peak: 7-9 AM
            - Evening peak: 5-7 PM
            - Weekends: Generally lighter traffic
            
            Provide predictions for:
            1. Next 30 minutes
            2. Next 2 hours
            3. Peak impact
            
            Format as JSON with: predicted_level, confidence, estimated_delay_minutes, 
            peak_hours, recommendations
            """
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=500,
                    response_mime_type="application/json"
                )
            )
            
            prediction = json.loads(response.text)
            prediction["predicted_at"] = datetime.now().isoformat()
            prediction["model"] = self.model
            
            return prediction
            
        except Exception as e:
            print(f"Traffic prediction error: {e}")
            return self.get_simulated_traffic_prediction(route_data)
    
    async def get_ai_alternatives(
        self,
        start_lat: float,
        start_lng: float,
        dest_lat: float,
        dest_lng: float,
        primary_route: Dict,
        realtime_data: Dict,
        optimize_for: str
    ) -> List[Dict]:
        """Find and analyze alternative routes with AI"""
        
        # Get alternative routes from OSRM
        alternatives = await self.get_osrm_alternatives(
            start_lat, start_lng, dest_lat, dest_lng
        )
        
        ai_alternatives = []
        
        for i, alt in enumerate(alternatives[:3]):  # Limit to 3 alternatives
            try:
                if self.client:
                    # Analyze each alternative with AI
                    analysis = await self.analyze_alternative_route(
                        alt, primary_route, realtime_data, optimize_for, i+1
                    )
                    ai_alternatives.append({
                        **alt,
                        "ai_analysis": analysis,
                        "ai_score": self.calculate_route_score(alt, optimize_for),
                        "recommendation": self.get_recommendation_level(alt, primary_route)
                    })
                else:
                    # Simulated analysis
                    ai_alternatives.append({
                        **alt,
                        "ai_analysis": self.simulate_route_analysis(alt, optimize_for),
                        "ai_score": random.uniform(0.6, 0.9),
                        "recommendation": random.choice(["good", "better", "best"])
                    })
                    
            except Exception as e:
                print(f"Alternative analysis error for route {i}: {e}")
                continue
        
        return ai_alternatives
    
    async def analyze_traffic_patterns(
        self,
        start_lat: float,
        start_lng: float,
        dest_lat: float,
        dest_lng: float,
        realtime_data: Dict
    ) -> Dict:
        """Analyze traffic patterns with AI"""
        try:
            prompt = f"""
            Analyze traffic patterns between these coordinates in Calapan City:
            Start: {start_lat}, {start_lng}
            Destination: {dest_lat}, {dest_lng}
            
            Current: {realtime_data}
            Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Provide analysis of:
            1. Typical traffic patterns for this route
            2. Peak hours to avoid
            3. Best times to travel
            4. Historical incident hotspots
            
            Format as JSON.
            """
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=800,
                    response_mime_type="application/json"
                )
            )
            
            return json.loads(response.text)
            
        except Exception as e:
            print(f"Traffic pattern analysis error: {e}")
            return {"error": str(e)}
    
    async def generate_recommendations(
        self,
        route_data: Dict,
        realtime_data: Dict,
        ai_insights: Dict,
        user_context: Optional[Dict] = None
    ) -> List[Dict]:
        """Generate personalized recommendations"""
        recommendations = []
        
        # Safety recommendations
        if ai_insights.get("safety_score", 7) < 7:
            recommendations.append({
                "type": "safety",
                "priority": "high",
                "message": "Consider safer alternative route",
                "icon": "🛡️"
            })
        
        # Traffic recommendations
        if realtime_data.get("traffic_level") == "high":
            recommendations.append({
                "type": "traffic",
                "priority": "medium",
                "message": "Heavy traffic expected - add 15-20 minutes",
                "icon": "🚦"
            })
        
        # Weather recommendations
        weather = realtime_data.get("weather", {})
        if weather.get("condition") in ["Rain", "Storm"]:
            recommendations.append({
                "type": "weather",
                "priority": "high",
                "message": "Poor weather conditions - drive carefully",
                "icon": "🌧️"
            })
        
        # Time-based recommendations
        hour = datetime.now().hour
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            recommendations.append({
                "type": "timing",
                "priority": "medium",
                "message": "Peak hours - consider waiting if possible",
                "icon": "⏰"
            })
        
        # Add AI-generated recommendations if available
        if self.client and ai_insights.get("recommendations"):
            ai_recs = ai_insights.get("recommendations", [])
            if isinstance(ai_recs, list):
                for rec in ai_recs[:3]:  # Limit to 3 AI recommendations
                    recommendations.append({
                        "type": "ai",
                        "priority": "low",
                        "message": rec,
                        "icon": "🤖"
                    })
        
        return recommendations
    
    # Helper methods
    async def get_osrm_route(self, start_lat, start_lng, dest_lat, dest_lng):
        """Get route from OSRM"""
        try:
            url = f"{os.getenv('OSRM_URL', 'http://router.project-osrm.org')}/route/v1/driving/{start_lng},{start_lat};{dest_lng},{dest_lat}?overview=full&geometries=geojson&alternatives=3"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if data.get("code") == "Ok":
                route = data["routes"][0]
                return {
                    "distance": route["distance"],
                    "duration": route["duration"],
                    "geometry": route["geometry"],
                    "legs": route.get("legs", []),
                    "confidence": 0.9
                }
        except Exception as e:
            print(f"OSRM error: {e}")
        
        # Fallback
        return {
            "distance": 5000,
            "duration": 600,
            "geometry": {"type": "LineString", "coordinates": []},
            "confidence": 0.5
        }
    
    async def get_osrm_alternatives(self, start_lat, start_lng, dest_lat, dest_lng):
        """Get alternative routes from OSRM"""
        try:
            url = f"{os.getenv('OSRM_URL', 'http://router.project-osrm.org')}/route/v1/driving/{start_lng},{start_lat};{dest_lng},{dest_lat}?overview=full&geometries=geojson&alternatives=true"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            alternatives = []
            if data.get("code") == "Ok":
                for i, route in enumerate(data["routes"][:3]):  # Get up to 3 alternatives
                    alternatives.append({
                        "id": f"alt_{i+1}",
                        "distance": route["distance"],
                        "duration": route["duration"],
                        "geometry": route["geometry"],
                        "summary": f"Alternative {i+1}",
                        "time_saving": 0  # Will be calculated
                    })
            return alternatives
        except:
            return []
    
    def calculate_ai_score(self, route, optimize_for):
        """Calculate AI score for route"""
        base_score = 0.7
        
        if optimize_for == "fastest":
            base_score += 0.1 if route["duration"] < 1800 else -0.1
        elif optimize_for == "safest":
            base_score += 0.15
        elif optimize_for == "scenic":
            base_score += 0.05
        
        return min(max(base_score + random.uniform(-0.1, 0.1), 0.5), 0.95)
    
    def calculate_time_savings(self, primary, alternatives):
        """Calculate time savings for alternatives"""
        savings = []
        primary_time = primary.get("duration", 0)
        
        for alt in alternatives:
            time_diff = primary_time - alt.get("duration", 0)
            if time_diff > 0:
                savings.append({
                    "alternative_id": alt.get("id"),
                    "minutes_saved": time_diff / 60,
                    "percentage": (time_diff / primary_time) * 100
                })
        
        return savings
    
    # Simulation methods (fallback when no API key)
    def get_simulated_insights(self, route_data, realtime_data, optimize_for):
        return {
            "safety_score": random.randint(6, 9),
            "emergency_access": "Good",
            "potential_delays": ["School zones during peak hours"],
            "weather_impact": "Minimal",
            "confidence": 0.7
        }
    
    def get_simulated_traffic_prediction(self, route_data):
        return {
            "predicted_level": random.choice(["low", "medium", "high"]),
            "confidence": random.uniform(0.6, 0.8),
            "estimated_delay_minutes": random.randint(0, 30),
            "peak_hours": ["7-9 AM", "5-7 PM"],
            "recommendations": ["Avoid peak hours if possible"]
        }
    
    async def get_simulated_incidents(self, start_lat, start_lng, dest_lat, dest_lng):
        incidents = []
        if random.random() > 0.7:  # 30% chance of incident
            incidents.append({
                "type": random.choice(["accident", "construction", "road_closed"]),
                "severity": random.choice(["low", "medium", "high"]),
                "location": "Near city center",
                "description": "Minor incident reported"
            })
        return incidents
    
    async def get_simulated_weather(self, lat, lng):
        conditions = ["Clear", "Partly Cloudy", "Cloudy", "Rain", "Storm"]
        return {
            "condition": random.choice(conditions),
            "temperature": random.randint(25, 35),
            "humidity": random.randint(60, 90),
            "visibility": random.choice(["Good", "Moderate", "Poor"])
        }
    
    def get_simulated_road_conditions(self):
        return {
            "surface": random.choice(["Good", "Fair", "Poor"]),
            "maintenance": random.choice(["Well-maintained", "Needs repair"]),
            "lighting": random.choice(["Good", "Adequate", "Poor"])
        }
    
    def get_fallback_realtime_data(self):
        return {
            "traffic_level": "medium",
            "average_speed": 40,
            "incidents": [],
            "weather": {"condition": "Clear", "temperature": 30},
            "road_conditions": {"surface": "Good"},
            "time_of_day": datetime.now().hour,
            "day_of_week": datetime.now().strftime("%A"),
            "last_updated": datetime.now().isoformat()
        }


# Create singleton instance
map_service = GeminiMapService()