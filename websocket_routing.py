# backend/websocket_routing.py
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List
import json
import asyncio
from datetime import datetime

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(websocket)
        
    def disconnect(self, websocket: WebSocket, user_id: str):
        if user_id in self.active_connections:
            self.active_connections[user_id].remove(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
                
    async def send_personal_message(self, message: dict, user_id: str):
        if user_id in self.active_connections:
            for connection in self.active_connections[user_id]:
                try:
                    await connection.send_json(message)
                except:
                    pass
                    
    async def broadcast(self, message: dict):
        for user_connections in self.active_connections.values():
            for connection in user_connections:
                try:
                    await connection.send_json(message)
                except:
                    pass

manager = ConnectionManager()

# Add this to main.py
from websocket_routing import manager

@app.websocket("/ws/route/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(websocket, user_id)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            # Handle different message types
            message_type = data.get('type')
            
            if message_type == 'position_update':
                # Update user position
                lat = data.get('lat')
                lng = data.get('lng')
                
                # Store position in real-time service
                realtime_service.update_user_position(user_id, lat, lng)
                
                # Send acknowledgment
                await manager.send_personal_message({
                    'type': 'position_updated',
                    'timestamp': datetime.now().isoformat(),
                    'position': {'lat': lat, 'lng': lng}
                }, user_id)
                
            elif message_type == 'subscribe_traffic':
                # Subscribe to traffic updates for a route
                route_id = data.get('route_id')
                
                # Start sending periodic traffic updates
                asyncio.create_task(
                    send_traffic_updates(user_id, route_id)
                )
                
            elif message_type == 'ping':
                await manager.send_personal_message({
                    'type': 'pong',
                    'timestamp': datetime.now().isoformat()
                }, user_id)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)

async def send_traffic_updates(user_id: str, route_id: str):
    """Send periodic traffic updates to subscribed users"""
    while True:
        try:
            # Check if user is still connected
            if user_id not in manager.active_connections:
                break
                
            # Get live traffic data (simulated)
            traffic_data = realtime_service.get_live_traffic(
                13.411, 121.181, 13.415, 121.201  # Example coordinates
            )
            
            if traffic_data['success']:
                await manager.send_personal_message({
                    'type': 'traffic_update',
                    'route_id': route_id,
                    'data': traffic_data['traffic_data'],
                    'timestamp': datetime.now().isoformat()
                }, user_id)
                
            # Wait before next update
            await asyncio.sleep(30)  # Update every 30 seconds
            
        except Exception as e:
            print(f"Traffic update error: {e}")
            break