# backend/websocket_manager.py
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List
import json
import asyncio
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        # Connected clients: {user_id: [websockets]}
        self.active_connections: Dict[str, List[WebSocket]] = {}
        # Real-time data stores
        self.user_positions: Dict[str, Dict] = {}
        self.route_updates: Dict[str, Dict] = {}
        self.traffic_updates: List[Dict] = []
        
    async def connect(self, websocket: WebSocket, user_id: str):
        """Accept new WebSocket connection"""
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(websocket)
        logger.info(f"User {user_id} connected. Total connections: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket, user_id: str):
        """Remove disconnected WebSocket"""
        if user_id in self.active_connections:
            self.active_connections[user_id].remove(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
            logger.info(f"User {user_id} disconnected. Remaining connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, user_id: str):
        """Send message to specific user"""
        if user_id in self.active_connections:
            for connection in self.active_connections[user_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Failed to send to user {user_id}: {e}")
    
    async def broadcast(self, message: dict):
        """Send message to all connected users"""
        for user_connections in self.active_connections.values():
            for connection in user_connections:
                try:
                    await connection.send_json(message)
                except:
                    pass
    
    def update_user_position(self, user_id: str, lat: float, lng: float, accuracy: float = None):
        """Update user's real-time position"""
        self.user_positions[user_id] = {
            'lat': lat,
            'lng': lng,
            'accuracy': accuracy,
            'timestamp': datetime.now().isoformat(),
            'updated_at': datetime.now()
        }
    
    def get_nearby_users(self, lat: float, lng: float, radius_km: float = 5) -> List[Dict]:
        """Get users within radius"""
        nearby = []
        for user_id, position in self.user_positions.items():
            # Simple distance calculation (in production, use Haversine)
            distance = abs(position['lat'] - lat) + abs(position['lng'] - lng)
            if distance <= radius_km / 111:  # Rough conversion: 1 degree ≈ 111km
                nearby.append({
                    'user_id': user_id,
                    'position': position,
                    'distance_approx_km': round(distance * 111, 2)
                })
        return nearby

# Global WebSocket manager instance
connection_manager = ConnectionManager()