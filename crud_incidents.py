# crud_incidents.py
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, func
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import json

from models import (
    IncidentReport, TrainingDataset, User, 
    IncidentCategory, BarangayInfo, EmergencyFacility, IncidentUpdate
)

def create_incident_report(
    db: Session,
    user_id: int,
    incident_data: dict,
    ml_analysis: dict,                # kept for compatibility, but we'll override with new params if provided
    image_paths: list = None,
    video_paths: list = None,
    text_analysis: dict = None,       # NEW: from predict_text()
    image_analysis: dict = None,      # NEW: from analyze_image()
    video_analysis: dict = None       # NEW: from analyze_video()
):
    import uuid
    import json
    from datetime import datetime

    report_id = str(uuid.uuid4())
    
    # ----- Determine which analysis to use -----
    # Prefer new structured params; fallback to old ml_analysis format
    if text_analysis is not None:
        incident_type = text_analysis.get("incident_type", "Other")
        severity = text_analysis.get("severity", "medium")
        confidence = text_analysis.get("type_confidence", 0.5)
        keywords = []   # zero‑shot doesn't extract keywords; optional
        full_text_json = json.dumps(text_analysis)
    else:
        # Old style: ml_analysis may contain "text" sub-dict or top-level fields
        if "text" in ml_analysis:
            text_part = ml_analysis["text"]
            incident_type = text_part.get("type", "Other")
            severity = text_part.get("severity", "medium")
            confidence = text_part.get("confidence", 0.5)
            keywords = text_part.get("keywords", [])
        else:
            incident_type = ml_analysis.get("type", "Other")
            severity = ml_analysis.get("severity", "medium")
            confidence = ml_analysis.get("confidence", 0.5)
            keywords = ml_analysis.get("keywords", [])
        full_text_json = json.dumps(ml_analysis)
    
    # Map severity to priority
    priority_map = {"low": 2, "medium": 3, "high": 4, "critical": 5}
    priority = priority_map.get(severity.lower(), 3)
    
    # Store image/video analyses as JSON (if provided)
    image_analysis_json = json.dumps(image_analysis) if image_analysis else None
    video_analysis_json = json.dumps(video_analysis) if video_analysis else None
    
    report = IncidentReport(
        id=report_id,
        user_id=user_id,
        description=incident_data.get("description"),
        incident_type=incident_type,
        severity=severity,
        priority=priority,
        ml_confidence=confidence,
        latitude=incident_data.get("latitude"),
        longitude=incident_data.get("longitude"),
        address=incident_data.get("address"),
        barangay=incident_data.get("barangay"),
        city="Calapan",
        province="Oriental Mindoro",
        contact_number=incident_data.get("contact_number"),
        emergency_contact=incident_data.get("emergency_contact"),
        image_paths=json.dumps(image_paths or []),
        video_paths=json.dumps(video_paths or []),
        text_analysis=full_text_json,
        image_analysis=image_analysis_json,   # NEW column
        video_analysis=video_analysis_json,   # NEW column
        keywords=json.dumps(keywords),
        status="pending",
        created_at=datetime.utcnow()
    )
    
    db.add(report)
    db.commit()
    db.refresh(report)
    return report

def save_to_training_dataset(db: Session, report_id: str, incident_data: Dict, ml_analysis: Dict):
    """Save incident data to training dataset"""
    
    training_data = TrainingDataset(
        report_id=report_id,
        description=incident_data.get("description"),
        incident_type=ml_analysis.get("incident_type", "Other"),
        severity=ml_analysis.get("severity", "Medium"),
        image_paths=json.dumps(ml_analysis.get("media_paths", {}).get("image", [])),
        video_paths=json.dumps(ml_analysis.get("media_paths", {}).get("video", [])),
        is_verified=False
    )
    
    db.add(training_data)
    db.commit()
    
    return training_data

def get_incident_report(db: Session, report_id: str):
    """Get incident report by ID"""
    return db.query(IncidentReport).filter(IncidentReport.id == report_id).first()

def get_user_incidents(db: Session, user_id: int, skip: int = 0, limit: int = 100):
    return db.query(IncidentReport).filter(IncidentReport.user_id == user_id).offset(skip).limit(limit).all()

def get_all_incidents(db: Session, skip: int = 0, limit: int = 100, 
                      status: Optional[str] = None, 
                      incident_type: Optional[str] = None,
                      barangay: Optional[str] = None):
    """Get all incidents with optional filters"""
    query = db.query(IncidentReport)
    
    if status:
        query = query.filter(IncidentReport.status == status)
    
    if incident_type:
        query = query.filter(IncidentReport.incident_type == incident_type)
    
    if barangay:
        query = query.filter(IncidentReport.barangay == barangay)
    
    return query.order_by(IncidentReport.created_at.desc())\
                .offset(skip)\
                .limit(limit)\
                .all()

def update_incident_status(db: Session, report_id: str, status: str, user_id: Optional[int] = None):
    """Update incident status"""
    incident = get_incident_report(db, report_id)
    if not incident:
        return None
    
    incident.status = status
    incident.updated_at = datetime.utcnow()
    
    if status == "verified" and user_id:
        incident.verified_by = user_id
    
    db.commit()
    db.refresh(incident)
    
    # Create update record
    create_incident_update(db, report_id, user_id, "status_change", 
                          f"Status changed to {status}")
    
    return incident

def assign_incident(db: Session, report_id: str, assignee_id: int, assigned_by: int):
    """Assign incident to a responder"""
    incident = get_incident_report(db, report_id)
    if not incident:
        return None
    
    assignee = db.query(User).filter(User.id == assignee_id).first()
    if not assignee:
        return None
    
    incident.assigned_to = assignee_id
    incident.status = "in_progress"
    incident.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(incident)
    
    # Create update record
    create_incident_update(db, report_id, assigned_by, "assignment", 
                          f"Assigned to {assignee.full_name}")
    
    return incident

def create_incident_update(db: Session, incident_id: str, user_id: int, 
                          update_type: str, content: str, metadata: Optional[Dict] = None):
    """Create an update for an incident"""
    update = IncidentUpdate(
        incident_id=incident_id,
        user_id=user_id,
        update_type=update_type,
        content=content,
        update_metadata=json.dumps(metadata) if metadata else None
    )
    
    db.add(update)
    db.commit()
    db.refresh(update)
    
    return update

def get_incident_updates(db: Session, incident_id: str):
    """Get all updates for an incident"""
    return db.query(IncidentUpdate)\
             .filter(IncidentUpdate.incident_id == incident_id)\
             .order_by(IncidentUpdate.created_at.asc())\
             .all()

def get_incident_categories(db: Session):
    """Get all incident categories"""
    return db.query(IncidentCategory).all()

def get_emergency_facilities(db: Session, barangay: Optional[str] = None, 
                            facility_type: Optional[str] = None):
    """Get emergency facilities with optional filters"""
    query = db.query(EmergencyFacility).filter(EmergencyFacility.is_available == True)
    
    if barangay:
        query = query.filter(EmergencyFacility.barangay == barangay)
    
    if facility_type:
        query = query.filter(EmergencyFacility.facility_type == facility_type)
    
    return query.all()

def search_incidents(db: Session, keyword: str = None, barangay: Optional[str] = None,
                    start_date: Optional[datetime] = None, end_date: Optional[datetime] = None):
    """Search incidents by keyword and filters"""
    query = db.query(IncidentReport)
    
    # Keyword search in description and keywords
    if keyword:
        query = query.filter(
            or_(
                IncidentReport.description.contains(keyword),
                IncidentReport.keywords.contains(keyword)
            )
        )
    
    if barangay:
        query = query.filter(IncidentReport.barangay == barangay)
    
    if start_date:
        query = query.filter(IncidentReport.created_at >= start_date)
    
    if end_date:
        query = query.filter(IncidentReport.created_at <= end_date)
    
    return query.order_by(IncidentReport.created_at.desc()).all()

def get_ml_stats_for_user(db: Session, user_id: int, days: int = 30):
    """
    Get ML-related stats for a specific user (optimized for dashboard)
    """
    cutoff = datetime.utcnow() - timedelta(days=days)
    
    stats_query = db.query(
        IncidentReport.incident_type.label('type'),
        IncidentReport.severity,
        IncidentReport.ml_confidence,
        func.count().label('count')
    ).filter(
        and_(
            IncidentReport.user_id == user_id,
            IncidentReport.created_at >= cutoff,
            IncidentReport.ml_confidence.isnot(None)
        )
    ).group_by(
        IncidentReport.incident_type,
        IncidentReport.severity,
        IncidentReport.ml_confidence
    ).all()
    
    # Aggregate results
    type_stats = {}
    severity_stats = {}
    confidences = []
    
    for row in stats_query:
        # Type stats
        t = row.type or 'Unknown'
        type_stats[t] = type_stats.get(t, 0) + row.count
        
        # Severity stats
        s = row.severity or 'Unknown'
        severity_stats[s] = severity_stats.get(s, 0) + row.count
        
        # Confidence tracking (average per incident)
        confidences.extend([row.ml_confidence] * row.count)
    
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    return {
        "by_type": type_stats,
        "by_severity": severity_stats,
        "avg_ml_confidence": round(avg_confidence, 3),
        "total_ml_analyzed": len(confidences),
        "days": days
    }

def get_statistics(db: Session, days: int = 30):
    """Get incident statistics for the last N days"""
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    # Total incidents
    total = db.query(IncidentReport)\
              .filter(IncidentReport.created_at >= start_date)\
              .count()
    
    # Incidents by type
    by_type_query = db.query(
        IncidentReport.incident_type,
        func.count(IncidentReport.id).label('count')
    ).filter(IncidentReport.created_at >= start_date)\
     .group_by(IncidentReport.incident_type)
    
    by_type = {row.incident_type: row.count for row in by_type_query.all()}
    
    # Incidents by status
    by_status_query = db.query(
        IncidentReport.status,
        func.count(IncidentReport.id).label('count')
    ).filter(IncidentReport.created_at >= start_date)\
     .group_by(IncidentReport.status)
    
    by_status = {row.status: row.count for row in by_status_query.all()}
    
    # Incidents by barangay
    by_barangay_query = db.query(
        IncidentReport.barangay,
        func.count(IncidentReport.id).label('count')
    ).filter(
        and_(
            IncidentReport.created_at >= start_date,
            IncidentReport.barangay.isnot(None)
        )
    ).group_by(IncidentReport.barangay)
    
    by_barangay = {row.barangay: row.count for row in by_barangay_query.all()}
    
    return {
        "total": total,
        "by_type": by_type,
        "by_status": by_status,
        "by_barangay": by_barangay,
        "time_period": f"Last {days} days"
    }

def get_barangays(db: Session):
    """Get all barangays"""
    return db.query(BarangayInfo).all()

def get_nearby_facilities(db: Session, lat: float, lng: float, radius_km: float = 5):
    """Get emergency facilities within radius of location"""
    # Simple approximation (for production, use proper spatial queries)
    facilities = db.query(EmergencyFacility)\
                   .filter(EmergencyFacility.is_available == True)\
                   .all()
    
    nearby = []
    for facility in facilities:
        # Calculate distance (simplified)
        distance = ((facility.latitude - lat) ** 2 + (facility.longitude - lng) ** 2) ** 0.5 * 111  # approx km
        if distance <= radius_km:
            facility_data = {
                "id": facility.id,
                "name": facility.name,
                "type": facility.facility_type,
                "distance_km": round(distance, 1),
                "contact": facility.contact_number,
                "address": facility.address,
                "barangay": facility.barangay
            }
            nearby.append(facility_data)
    
    # Sort by distance
    nearby.sort(key=lambda x: x["distance_km"])
    return nearby