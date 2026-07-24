import requests
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class RoutingService:
    def __init__(self):
        self.base_url = "http://router.project-osrm.org/route/v1"
    
    def calculate_route(
        self, 
        start_lat: float, 
        start_lng: float, 
        dest_lat: float, 
        dest_lng: float,
        profile: str = "driving"      # ← NEW: profile parameter
    ) -> Dict[str, Any]:
        """
        Calculate route between two points for any OSRM profile.
        Supported profiles: driving, foot, bike, etc.
        """
        try:
            # OSRM expects lng,lat order
            url = f"{self.base_url}/{profile}/{start_lng},{start_lat};{dest_lng},{dest_lat}"
            
            params = {
                "overview": "full",
                "geometries": "geojson",
                "steps": "true",
                "annotations": "true"
            }
            
            logger.info(f"Requesting {profile} route from OSRM: {url}")
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("code") == "Ok" and data.get("routes"):
                    route = data["routes"][0]
                    return {
                        "success": True,
                        "geometry": route["geometry"],
                        "distance": route["distance"],
                        "duration": route["duration"],
                        "waypoints": data.get("waypoints", []),
                        "legs": route.get("legs", []),
                        "instructions": self._extract_instructions(route),
                        "routes": [{
                            "distance": route["distance"],
                            "duration": route["duration"],
                            "traffic_delay": 0
                        }]
                    }
                else:
                    return {"success": False, "error": "No route found"}
            else:
                return {"success": False, "error": f"OSRM error: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Routing error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _extract_instructions(self, route_data: Dict) -> list:
        instructions = []
        for leg in route_data.get("legs", []):
            for step in leg.get("steps", []):
                instruction = {
                    "distance": step.get("distance", 0),
                    "duration": step.get("duration", 0),
                    "type": step.get("maneuver", {}).get("type", "continue"),
                    "instruction": step.get("name", "Continue"),
                    "way_points": step.get("way_points", [])
                }
                instructions.append(instruction)
        return instructions