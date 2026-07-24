# backend/realtime_routing_service.py
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import requests
from collections import defaultdict

logger = logging.getLogger(__name__)

class RealTimeRoutingService:
    """Real-time routing with live traffic, alerts, and alternative routes"""
    
    def __init__(self):
        self.osrm_url = "http://router.project-osrm.org/route/v1"
        self.traffic_data = self._initialize_traffic_data()
        self.alerts = self._initialize_alerts()
        self.user_positions = {}
        
    def _initialize_traffic_data(self) -> Dict:
        """Initialize mock traffic data for Calapan City"""
        return {
            'segments': {
                'highway': {'level': 'low', 'speed': 80},
                'downtown': {'level': 'high', 'speed': 20},
                'market_area': {'level': 'medium', 'speed': 40},
                'port_area': {'level': 'medium', 'speed': 50},
                'school_zones': {'level': 'high', 'speed': 15}
            },
            'last_updated': datetime.now(),
            'update_interval': 30  # seconds
        }
    
    def _initialize_alerts(self) -> List[Dict]:
        """Initialize mock traffic alerts"""
        return [
            {
                'id': 1,
                'type': 'accident',
                'message': 'Accident reported on National Highway near Calapan Port',
                'severity': 'high',
                'location': {'lat': 13.415, 'lng': 121.201},
                'radius': 500,  # meters
                'start_time': datetime.now() - timedelta(minutes=15),
                'end_time': datetime.now() + timedelta(hours=1)
            },
            {
                'id': 2,
                'type': 'road_closure',
                'message': 'Road maintenance on M.H. del Pilar Street',
                'severity': 'medium',
                'location': {'lat': 13.408, 'lng': 121.189},
                'radius': 300,
                'start_time': datetime.now() - timedelta(minutes=30),
                'end_time': datetime.now() + timedelta(hours=2)
            },
            {
                'id': 3,
                'type': 'traffic_jam',
                'message': 'Heavy traffic near Calapan Public Market',
                'severity': 'high',
                'location': {'lat': 13.408, 'lng': 121.189},
                'radius': 400,
                'start_time': datetime.now() - timedelta(minutes=45),
                'end_time': None  # Ongoing
            }
        ]
    
    def get_live_traffic(
        self,
        start_lat: float,
        start_lng: float,
        dest_lat: float,
        dest_lng: float
    ) -> Dict[str, Any]:
        """
        Get live traffic data for a route
        
        Returns: {
            'traffic_level': 'low'/'medium'/'high',
            'average_speed': km/h,
            'estimated_delay': seconds,
            'congestion_points': [...]
        }
        """
        try:
            # Simulate real-time traffic data
            current_time = datetime.now()
            hour = current_time.hour
            
            # Determine traffic level based on time of day
            traffic_level = 'low'
            average_speed = 60  # km/h
            
            if 7 <= hour <= 9 or 16 <= hour <= 19:  # Peak hours
                traffic_level = 'high'
                average_speed = 20
            elif 10 <= hour <= 15:  # Midday
                traffic_level = 'medium'
                average_speed = 40
            elif 20 <= hour <= 22:  # Evening
                traffic_level = 'medium'
                average_speed = 50
            
            # Add randomness
            random_factor = random.uniform(0.8, 1.2)
            average_speed *= random_factor
            
            # Calculate estimated delay (simplified)
            route_length = self._estimate_distance(start_lat, start_lng, dest_lat, dest_lng)
            optimal_time = (route_length / 60) * 3600  # Assuming 60 km/h optimal
            current_time_estimate = (route_length / average_speed) * 3600
            estimated_delay = max(0, current_time_estimate - optimal_time)
            
            # Get congestion points
            congestion_points = self._get_congestion_points(
                start_lat, start_lng, dest_lat, dest_lng
            )
            
            # Get relevant alerts
            relevant_alerts = self._get_route_alerts(
                start_lat, start_lng, dest_lat, dest_lng
            )
            
            return {
                'success': True,
                'traffic_data': {
                    'traffic_level': traffic_level,
                    'average_speed': round(average_speed, 1),
                    'estimated_delay': round(estimated_delay),
                    'updated_at': current_time.isoformat(),
                    'congestion_points': congestion_points,
                    'route_length_km': round(route_length, 1)
                },
                'alerts': relevant_alerts
            }
            
        except Exception as e:
            logger.error(f"Live traffic error: {str(e)}")
            return {
                'success': False,
                'error': f"Live traffic service unavailable: {str(e)}"
            }
    
    def _estimate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Estimate distance between two points (simplified)"""
        # Haversine formula (simplified)
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371  # Earth radius in km
        
        lat1_rad = radians(lat1)
        lat2_rad = radians(lat2)
        delta_lat = radians(lat2 - lat1)
        delta_lng = radians(lng2 - lng1)
        
        a = sin(delta_lat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lng/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        distance = R * c
        
        # Add route factor (real routes are longer than straight line)
        return distance * 1.3
    
    def _get_congestion_points(
        self,
        start_lat: float,
        start_lng: float,
        dest_lat: float,
        dest_lng: float
    ) -> List[Dict]:
        """Get congestion points along the route"""
        # Simulate congestion points based on known problem areas
        congestion_areas = [
            {'lat': 13.411, 'lng': 121.181, 'name': 'City Hall', 'severity': 'high'},
            {'lat': 13.415, 'lng': 121.201, 'name': 'Port Area', 'severity': 'medium'},
            {'lat': 13.408, 'lng': 121.189, 'name': 'Public Market', 'severity': 'high'},
            {'lat': 13.418, 'lng': 121.195, 'name': 'SM City', 'severity': 'medium'},
        ]
        
        # Filter points that are roughly along the route
        return [
            point for point in congestion_areas
            if self._is_point_near_route(
                point['lat'], point['lng'],
                start_lat, start_lng, dest_lat, dest_lng,
                threshold_km=2  # Within 2km of route
            )
        ]
    
    def _is_point_near_route(
        self,
        point_lat: float,
        point_lng: float,
        start_lat: float,
        start_lng: float,
        dest_lat: float,
        dest_lng: float,
        threshold_km: float = 2
    ) -> bool:
        """Check if a point is near the route line"""
        # Simplified distance to line segment check
        from math import radians, sin, cos, sqrt, atan2
        
        # Convert to radians
        lat1, lon1 = radians(start_lat), radians(start_lng)
        lat2, lon2 = radians(dest_lat), radians(dest_lng)
        lat3, lon3 = radians(point_lat), radians(point_lng)
        
        # Calculate distance from point to line segment
        # Using spherical earth approximation
        R = 6371
        
        # Distance between start and destination
        d12 = 2 * asin(sqrt(sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2))
        
        # Distance from point to start
        d13 = 2 * asin(sqrt(sin((lat3-lat1)/2)**2 + cos(lat1)*cos(lat3)*sin((lon3-lon1)/2)**2))
        
        # Distance from point to destination
        d23 = 2 * asin(sqrt(sin((lat3-lat2)/2)**2 + cos(lat2)*cos(lat3)*sin((lon3-lon2)/2)**2))
        
        # Calculate cross-track distance
        if d12 == 0:
            return d13 * R <= threshold_km
        
        # Angular distance along great circle
        theta13 = d13 / d12
        theta23 = d23 / d12
        
        # Simplified check - if point is between start and end
        if 0 <= theta13 <= 1 or 0 <= theta23 <= 1:
            # Calculate perpendicular distance
            a = sin(d13) * sin(d23)
            b = sqrt(sin(d13)**2 - a**2)
            distance = b * R
            
            return distance <= threshold_km
        
        return False
    
    def _get_route_alerts(
        self,
        start_lat: float,
        start_lng: float,
        dest_lat: float,
        dest_lng: float
    ) -> List[Dict]:
        """Get traffic alerts that affect the route"""
        current_time = datetime.now()
        relevant_alerts = []
        
        for alert in self.alerts:
            # Check if alert is still active
            if alert['end_time'] and current_time > alert['end_time']:
                continue
            
            # Check if alert is near the route
            alert_lat = alert['location']['lat']
            alert_lng = alert['location']['lng']
            
            if self._is_point_near_route(
                alert_lat, alert_lng,
                start_lat, start_lng, dest_lat, dest_lng,
                threshold_km=alert['radius'] / 1000  # Convert meters to km
            ):
                relevant_alerts.append({
                    'id': alert['id'],
                    'type': alert['type'],
                    'message': alert['message'],
                    'severity': alert['severity'],
                    'location': alert['location'],
                    'distance_from_route': self._calculate_distance(
                        alert_lat, alert_lng,
                        (start_lat + dest_lat) / 2, (start_lng + dest_lng) / 2
                    )
                })
        
        return relevant_alerts
    
    def get_alternative_routes(
        self,
        start_lat: float,
        start_lng: float,
        dest_lat: float,
        dest_lng: float,
        current_route_time: float = 0
    ) -> Dict[str, Any]:
        """
        Find better alternative routes
        
        Returns: {
            'alternatives': [...],
            'best_alternative': {...},
            'time_savings': {...}
        }
        """
        try:
            # Get multiple route alternatives from OSRM
            url = f"{self.osrm_url}/driving/{start_lng},{start_lat};{dest_lng},{dest_lat}"
            
            params = {
                "overview": "simplified",
                "geometries": "geojson",
                "alternatives": "3",  # Get up to 3 alternatives
                "steps": "false"
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("code") == "Ok" and data.get("routes"):
                    routes = data["routes"]
                    
                    # Analyze alternatives
                    alternatives = []
                    for idx, route in enumerate(routes):
                        # Adjust duration based on current traffic
                        adjusted_duration = self._adjust_duration_for_traffic(
                            route.get('duration', 0),
                            start_lat, start_lng, dest_lat, dest_lng
                        )
                        
                        alternative = {
                            'index': idx,
                            'distance': route.get('distance', 0),
                            'original_duration': route.get('duration', 0),
                            'adjusted_duration': adjusted_duration,
                            'geometry': route.get('geometry'),
                            'summary': self._generate_route_summary(route)
                        }
                        
                        # Calculate time saving compared to current route
                        if current_route_time > 0:
                            alternative['time_saving'] = current_route_time - adjusted_duration
                        
                        alternatives.append(alternative)
                    
                    # Sort by adjusted duration (fastest first)
                    alternatives.sort(key=lambda x: x['adjusted_duration'])
                    
                    # Find best alternative (if better than current)
                    best_alternative = None
                    if current_route_time > 0 and alternatives:
                        for alt in alternatives:
                            if alt['time_saving'] > 60:  # Save at least 1 minute
                                best_alternative = alt
                                break
                    
                    return {
                        'success': True,
                        'alternatives': alternatives[:3],  # Top 3
                        'best_alternative': best_alternative,
                        'total_alternatives_found': len(alternatives)
                    }
            
            return {
                'success': False,
                'error': 'Could not find alternative routes'
            }
            
        except Exception as e:
            logger.error(f"Alternative routes error: {str(e)}")
            return {
                'success': False,
                'error': f"Alternative route service error: {str(e)}"
            }
    
    def _adjust_duration_for_traffic(
        self,
        base_duration: float,
        start_lat: float,
        start_lng: float,
        dest_lat: float,
        dest_lng: float
    ) -> float:
        """Adjust route duration based on current traffic"""
        traffic_data = self.get_live_traffic(
            start_lat, start_lng, dest_lat, dest_lng
        )
        
        if traffic_data['success']:
            traffic_level = traffic_data['traffic_data']['traffic_level']
            
            # Traffic multipliers
            multipliers = {
                'low': 1.0,      # No delay
                'medium': 1.3,   # 30% longer
                'high': 1.8      # 80% longer
            }
            
            multiplier = multipliers.get(traffic_level, 1.0)
            return base_duration * multiplier
        
        return base_duration
    
    def _generate_route_summary(self, route: Dict) -> str:
        """Generate human-readable route summary"""
        distance_km = route.get('distance', 0) / 1000
        duration_min = route.get('duration', 0) / 60
        
        if distance_km < 2:
            return f"Short route ({distance_km:.1f}km, {duration_min:.0f}min)"
        elif distance_km < 5:
            return f"Medium route ({distance_km:.1f}km, {duration_min:.0f}min)"
        else:
            return f"Long route ({distance_km:.1f}km, {duration_min:.0f}min)"
    
    def calculate_current_eta(
        self,
        start_lat: float,
        start_lng: float,
        dest_lat: float,
        dest_lng: float,
        original_duration: float,
        traffic_level: str = None
    ) -> Dict[str, Any]:
        """
        Calculate current ETA with live traffic
        
        Returns: {
            'current_eta': seconds,
            'delay': seconds,
            'traffic_level': str,
            'updated_at': ISO timestamp
        }
        """
        try:
            # Get current traffic if not provided
            if not traffic_level:
                traffic_data = self.get_live_traffic(
                    start_lat, start_lng, dest_lat, dest_lng
                )
                if traffic_data['success']:
                    traffic_level = traffic_data['traffic_data']['traffic_level']
                else:
                    traffic_level = 'medium'
            
            # Calculate delay based on traffic level
            delays = {
                'low': 0.0,      # 0% delay
                'medium': 0.3,   # 30% delay
                'high': 0.8      # 80% delay
            }
            
            delay_multiplier = delays.get(traffic_level, 0.3)
            current_eta = original_duration * (1 + delay_multiplier)
            delay = current_eta - original_duration
            
            return {
                'success': True,
                'current_eta': round(current_eta),
                'delay': round(delay),
                'traffic_level': traffic_level,
                'updated_at': datetime.now().isoformat(),
                'original_duration': original_duration,
                'delay_percentage': round(delay_multiplier * 100)
            }
            
        except Exception as e:
            logger.error(f"ETA calculation error: {str(e)}")
            return {
                'success': False,
                'error': f"ETA calculation failed: {str(e)}",
                'current_eta': original_duration,
                'delay': 0
            }
    
    def get_traffic_alerts(
        self,
        start_lat: float,
        start_lng: float,
        dest_lat: float,
        dest_lng: float,
        radius_km: float = 5
    ) -> Dict[str, Any]:
        """
        Get all traffic alerts near a route
        
        Returns: {
            'alerts': [...],
            'high_severity_count': int,
            'recent_alerts': [...]
        }
        """
        try:
            current_time = datetime.now()
            
            # Filter active alerts
            active_alerts = []
            for alert in self.alerts:
                # Check if alert is still active
                if alert['end_time'] and current_time > alert['end_time']:
                    continue
                
                # Check if alert is within radius of route midpoint
                midpoint_lat = (start_lat + dest_lat) / 2
                midpoint_lng = (start_lng + dest_lng) / 2
                
                alert_lat = alert['location']['lat']
                alert_lng = alert['location']['lng']
                
                distance = self._calculate_distance(
                    alert_lat, alert_lng,
                    midpoint_lat, midpoint_lng
                )
                
                if distance <= radius_km:
                    # Calculate time since alert
                    time_since = current_time - alert['start_time']
                    hours_since = time_since.total_seconds() / 3600
                    
                    active_alerts.append({
                        **alert,
                        'distance_km': round(distance, 2),
                        'hours_since': round(hours_since, 1),
                        'is_recent': hours_since < 1  # Within last hour
                    })
            
            # Count by severity
            severity_counts = defaultdict(int)
            for alert in active_alerts:
                severity_counts[alert['severity']] += 1
            
            # Get recent alerts (last hour)
            recent_alerts = [a for a in active_alerts if a['is_recent']]
            
            return {
                'success': True,
                'alerts': active_alerts,
                'high_severity_count': severity_counts.get('high', 0),
                'medium_severity_count': severity_counts.get('medium', 0),
                'low_severity_count': severity_counts.get('low', 0),
                'recent_alerts': recent_alerts,
                'total_alerts': len(active_alerts),
                'last_updated': current_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Traffic alerts error: {str(e)}")
            return {
                'success': False,
                'error': f"Could not fetch traffic alerts: {str(e)}",
                'alerts': []
            }
    
    def _calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate distance between two points in kilometers"""
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371  # Earth's radius in km
        
        lat1_rad = radians(lat1)
        lat2_rad = radians(lat2)
        delta_lat = radians(lat2 - lat1)
        delta_lng = radians(lng2 - lng1)
        
        a = sin(delta_lat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lng/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def update_user_position(
        self,
        user_id: str,
        lat: float,
        lng: float,
        accuracy: float = None
    ) -> Dict[str, Any]:
        """Update real-time user position for tracking"""
        self.user_positions[user_id] = {
            'lat': lat,
            'lng': lng,
            'accuracy': accuracy,
            'timestamp': datetime.now().isoformat(),
            'last_updated': datetime.now()
        }
        
        return {
            'success': True,
            'user_id': user_id,
            'position': self.user_positions[user_id],
            'total_users_tracking': len(self.user_positions)
        }
    
    def get_nearby_users(
        self,
        lat: float,
        lng: float,
        radius_km: float = 5
    ) -> List[Dict]:
        """Get nearby users for social features"""
        nearby = []
        
        for user_id, position in self.user_positions.items():
            # Check if position is recent (last 5 minutes)
            last_updated = position['last_updated']
            if (datetime.now() - last_updated).total_seconds() > 300:
                continue
            
            distance = self._calculate_distance(
                lat, lng,
                position['lat'], position['lng']
            )
            
            if distance <= radius_km:
                nearby.append({
                    'user_id': user_id,
                    'distance_km': round(distance, 2),
                    'position': position,
                    'is_nearby': distance <= 1  # Within 1km
                })
        
        return nearby