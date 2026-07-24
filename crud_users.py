from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, or_
from datetime import datetime, timedelta
from models import User
from security import hash_password, verify_password
from schemas import RegisterIn
import logging
from sqlalchemy import or_, and_
from typing import Optional, Dict, Any
from security import hash_password, verify_password

def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email.lower()).first()

def create_user_admin(db: Session, user_data: Dict[str, Any]) -> User:
    """
    Create a new user (admin function)
    """
    # Check if password is provided in the data
    if "password" in user_data:
        password = user_data.pop("password")
        password_hash = hash_password(password)
    else:
        raise ValueError("Password is required")
    
    # Create user object
    user = User(
        full_name=user_data.get("full_name"),
        email=user_data.get("email", "").lower(),
        password_hash=password_hash,
        role=user_data.get("role", "user"),
        status=user_data.get("status", "active"),
        contact_number=user_data.get("contact_number"),
        barangay=user_data.get("barangay"),
        address=user_data.get("address"),
        emergency_contact_name=user_data.get("emergency_contact_name"),
        emergency_contact_number=user_data.get("emergency_contact_number")
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def authenticate_user(db: Session, email: str, password: str):
    user = get_user_by_email(db, email)
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user

def get_user_by_id(db: Session, user_id: int):
    return db.query(User).filter(User.id == user_id).first()


def update_user_profile(db: Session, user, data):
    for key, value in data.dict(exclude_unset=True).items():
        setattr(user, key, value)

    db.commit()
    db.refresh(user)
    return user

# Add these functions to your existing crud_users.py

def get_all_users(db: Session, skip: int = 0, limit: int = 100, role: Optional[str] = None, search: Optional[str] = None):
    """
    Get all users with optional filtering, including online status.
    """
    query = db.query(User)
    
    # Apply filters
    if role:
        query = query.filter(User.role == role)
    
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            (User.full_name.ilike(search_term)) |
            (User.email.ilike(search_term)) |
            (User.contact_number.ilike(search_term))
        )
    
    total = query.count()
    users = query.order_by(User.created_at.desc()).offset(skip).limit(limit).all()
    
    # Compute online status (active within last 3 minutes)
    now = datetime.utcnow()
    threshold = now - timedelta(minutes=3)
    
    for user in users:
        user.is_online = user.last_active is not None and user.last_active > threshold
        # Optionally, you can also attach last_active for frontend use
        # (already available via user.last_active)
    
    return {
        "users": users,
        "total": total,
        "page": (skip // limit) + 1,
        "limit": limit,
        "total_pages": (total + limit - 1) // limit if limit > 0 else 0
    }


def update_user_role(db: Session, user_id: int, role: str):
    """
    Update user role (admin only)
    """
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.role = role
        db.commit()
        db.refresh(user)
    return user

def get_user_by_id(db: Session, user_id: int):
    """
    Get user by ID
    """
    return db.query(User).filter(User.id == user_id).first()

def get_user_statistics(db: Session):
    """
    Get user statistics for admin dashboard
    """
    from sqlalchemy import func
    
    # Total users
    total_users = db.query(func.count(User.id)).scalar()
    
    # Users by role
    role_stats = db.query(User.role, func.count(User.id)).group_by(User.role).all()
    
    # New users this month
    from datetime import datetime, timedelta
    start_of_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    new_users_month = db.query(func.count(User.id)).filter(User.created_at >= start_of_month).scalar()
    
    # Convert role_stats to dictionary
    role_dict = {role: count for role, count in role_stats}
    
    return {
        "total_users": total_users,
        "by_role": role_dict,
        "new_users_this_month": new_users_month,
        "citizens": role_dict.get("user", 0),
        "responders": role_dict.get("responder", 0),
        "administrators": role_dict.get("admin", 0)
    }

def search_users(db: Session, search_term: str):
    """
    Search users by name, email, or contact number
    """
    search_pattern = f"%{search_term}%"
    return db.query(User).filter(
        (User.full_name.ilike(search_pattern)) |
        (User.email.ilike(search_pattern)) |
        (User.contact_number.ilike(search_pattern))
    ).all()

from sqlalchemy import or_, and_
from typing import Optional, Dict, Any

def get_all_users(
    db: Session, 
    skip: int = 0, 
    limit: int = 100, 
    role: Optional[str] = None,
    status: Optional[str] = None,
    barangay: Optional[str] = None,
    search: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get all users with optional filtering
    """
    query = db.query(User)
    
    # Apply filters
    if role and role != 'all':
        query = query.filter(User.role == role)
    
    if status and status != 'all':
        query = query.filter(User.status == status)
    
    if barangay and barangay != 'all':
        query = query.filter(User.barangay == barangay)
    
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            or_(
                User.full_name.ilike(search_term),
                User.email.ilike(search_term),
                User.contact_number.ilike(search_term),
                User.barangay.ilike(search_term)
            )
        )
    
    # Get total count
    total = query.count()
    
    # Apply pagination
    users = query.order_by(User.created_at.desc()).offset(skip).limit(limit).all()
    
    # Calculate pagination info
    total_pages = (total + limit - 1) // limit if limit > 0 else 0
    current_page = (skip // limit) + 1 if limit > 0 else 1
    
    return {
        "users": users,
        "total": total,
        "page": current_page,
        "limit": limit,
        "total_pages": total_pages
    }

def update_user_role(db: Session, user_id: int, role: str) -> Optional[User]:
    """
    Update user role
    """
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.role = role
        db.commit()
        db.refresh(user)
    return user

def update_user_status(db: Session, user_id: int, status: str) -> Optional[User]:
    """
    Update user status (active/inactive)
    """
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.status = status
        db.commit()
        db.refresh(user)
    return user

def get_user_statistics(db: Session) -> Dict[str, Any]:
    """
    Get user statistics for admin dashboard
    """
    from sqlalchemy import func
    
    # Total users
    total_users = db.query(func.count(User.id)).scalar() or 0
    
    # Users by role
    role_stats = db.query(User.role, func.count(User.id)).group_by(User.role).all()
    
    # Users by status
    status_stats = db.query(User.status, func.count(User.id)).group_by(User.status).all()
    
    # Convert to dictionaries
    role_dict = {role: count for role, count in role_stats}
    status_dict = {status: count for status, count in status_stats}
    
    return {
        "total_users": total_users,
        "by_role": role_dict,
        "by_status": status_dict,
        "citizens": role_dict.get("user", 0),
        "responders": role_dict.get("responder", 0),
        "tmoOfficers": role_dict.get("tmo", 0),
        "administrators": role_dict.get("admin", 0),
        "active_users": status_dict.get("active", 0),
        "inactive_users": status_dict.get("inactive", 0)
    }

def create_user_admin(db: Session, user_data: Dict[str, Any]) -> User:
    """
    Create a new user (admin function)
    """
    # Check if password is provided, otherwise use a default
    if "password" in user_data:
        password_hash = hash_password(user_data["password"])
    else:
        # Generate a random password if not provided
        import secrets
        random_password = secrets.token_urlsafe(12)
        password_hash = hash_password(random_password)
    
    # Create user object
    user = User(
        full_name=user_data["full_name"],
        email=user_data["email"].lower(),
        password_hash=password_hash,
        role=user_data.get("role", "user"),
        status=user_data.get("status", "active"),
        contact_number=user_data.get("contact_number"),
        barangay=user_data.get("barangay"),
        address=user_data.get("address"),
        emergency_contact_name=user_data.get("emergency_contact_name"),
        emergency_contact_number=user_data.get("emergency_contact_number")
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def delete_user(db: Session, user_id: int) -> bool:
    """
    Delete a user
    """
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        db.delete(user)
        db.commit()
        return True
    return False

def create_user(db: Session, user_data: RegisterIn) -> User:
    """
    Create a new regular user from registration data.
    """
    # Hash the password before storing
    password_hash = hash_password(user_data.password)

    # Create the user object
    user = User(
        full_name=user_data.full_name,
        email=user_data.email.lower(),
        password_hash=password_hash,
        role="user",                     # default role for self‑registration
        status="active",
        contact_number=user_data.contact_number,
        barangay=user_data.barangay,
        address=user_data.address,
        emergency_contact_name=user_data.emergency_contact_name,
        emergency_contact_number=user_data.emergency_contact_number
    )

    db.add(user)
    db.commit()
    db.refresh(user)
    return user