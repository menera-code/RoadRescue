import os
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime

from deps import get_db, get_current_user
from models import User, ChatHistory
from schemas import ChatRequest, ChatResponse, ChatHistoryItem
from gemini_client import chatbot

router = APIRouter()

def get_user_context(user: User) -> dict:
    """Get user context for AI personalization"""
    return {
        "user_id": user.id,
        "full_name": user.full_name,
        "role": user.role,
        "barangay": user.barangay,
        "contact_number": user.contact_number,
        "emergency_contact": {
            "name": user.emergency_contact_name,
            "number": user.emergency_contact_number
        }
    }

# Remove the middleware - it's not needed here since we have it in main.py
# @router.middleware("http")
# async def add_cors_headers(request, call_next):
#     response = await call_next(request)
#     response.headers["Access-Control-Allow-Origin"] = "*"
#     response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
#     response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
#     return response

def generate_quick_replies(user_message: str, ai_response: str) -> List[str]:
    """Generate context-aware quick reply suggestions"""
    user_msg_lower = user_message.lower()
    
    quick_replies = []
    
    # Based on user query
    if any(word in user_msg_lower for word in ["report", "incident", "accident"]):
        quick_replies.extend([
            "How to report a fire?",
            "What details are needed for a report?",
            "Can I upload photos with my report?"
        ])
    
    elif any(word in user_msg_lower for word in ["emergency", "help", "urgent"]):
        quick_replies.extend([
            "Emergency contact numbers",
            "What to do in a medical emergency?",
            "How to share my location?"
        ])
    
    elif any(word in user_msg_lower for word in ["traffic", "road"]):
        quick_replies.extend([
            "Current traffic conditions",
            "Road closure information",
            "Alternative routes"
        ])
    
    # Default suggestions
    else:
        quick_replies = [
            "Report an incident",
            "Emergency procedures",
            "Traffic updates"
        ]
    
    return quick_replies[:3]

@router.post("/chat/message", response_model=ChatResponse)
async def send_message(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Send a message to the AI chatbot"""
    try:
        print(f"DEBUG: Chat message from user {current_user.id}: {request.message}")
        
        # Save user message to database
        user_msg = ChatHistory(
            user_id=current_user.id,
            role="user",
            message=request.message
        )
        db.add(user_msg)
        db.commit()
        
        # Prepare conversation history for AI
        ai_messages = []
        if request.conversation_history:
            # Use provided history
            for msg in request.conversation_history[-10:]:  # Last 10 messages for context
                ai_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        else:
            # Get recent history from DB
            recent_msgs = db.query(ChatHistory).filter(
                ChatHistory.user_id == current_user.id
            ).order_by(ChatHistory.created_at.desc()).limit(5).all()
            
            for msg in reversed(recent_msgs):  # Oldest to newest
                ai_messages.append({
                    "role": msg.role,
                    "content": msg.message
                })
        
        # Add current user message
        ai_messages.append({
            "role": "user",
            "content": request.message
        })
        
        # Get AI response
        ai_response_content = await chatbot.get_response(ai_messages)
        
        # Save AI response to database
        ai_msg = ChatHistory(
            user_id=current_user.id,
            role="assistant",
            message=ai_response_content
        )
        db.add(ai_msg)
        db.commit()
        
        # Generate quick replies based on context
        quick_replies = generate_quick_replies(request.message, ai_response_content)
        
        return ChatResponse(
            role="assistant",
            content=ai_response_content,
            timestamp=datetime.utcnow().isoformat(),
            quick_replies=quick_replies
        )
        
    except Exception as e:
        print(f"ERROR in send_message: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ========== KEEP ONLY ONE get_chat_history FUNCTION ==========
@router.get("/chat/history", response_model=List[ChatHistoryItem])
async def get_chat_history(
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get user's chat history"""
    try:
        print(f"DEBUG: Getting chat history for user {current_user.id}")
        
        # For now, return empty array to avoid database errors
        # Check if table exists first
        from sqlalchemy import inspect
        inspector = inspect(db.bind)
        
        if 'chat_history' not in inspector.get_table_names():
            print("WARNING: chat_history table doesn't exist yet")
            return []
        
        messages = db.query(ChatHistory).filter(
            ChatHistory.user_id == current_user.id
        ).order_by(ChatHistory.created_at.asc()).limit(limit).all()
        
        print(f"DEBUG: Found {len(messages)} messages")
        
        formatted_messages = []
        for msg in messages:
            formatted_messages.append(
                ChatHistoryItem(
                    role=msg.role,
                    content=msg.message,
                    timestamp=msg.created_at.isoformat()
                )
            )
        
        return formatted_messages
        
    except Exception as e:
        print(f"ERROR in get_chat_history: {e}")
        import traceback
        traceback.print_exc()
        # Return empty array even on error
        return []

@router.delete("/chat/history")
async def clear_chat_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Clear user's chat history"""
    try:
        print(f"DEBUG: Clearing chat history for user {current_user.id}")
        
        deleted_count = db.query(ChatHistory).filter(
            ChatHistory.user_id == current_user.id
        ).delete()
        
        db.commit()
        
        return {
            "message": f"Cleared {deleted_count} messages from chat history",
            "deleted_count": deleted_count
        }
        
    except Exception as e:
        print(f"ERROR in clear_chat_history: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))