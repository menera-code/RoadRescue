from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Float, Boolean, ForeignKey
from datetime import datetime
from database import Base
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, Float, String, DateTime
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, Text, JSON, Enum as SQLAlchemyEnum
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
USER_ROLES = ("user", "admin", "responder")

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(150), nullable=False)
    email = Column(String(150), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), default="user")
    status = Column(String(20), default="active")  # active, inactive, suspended
    contact_number = Column(String(20))
    barangay = Column(String(100))
    address = Column(String(255))
    emergency_contact_name = Column(String(150))
    emergency_contact_number = Column(String(20))
    profile_photo = Column(String(255), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, nullable=True)   # <-- add this
    is_online = Column(Boolean, default=False)   # ✅ add this
    
    # Relationships
    reported_incidents = relationship("IncidentReport", back_populates="reporter", foreign_keys="IncidentReport.user_id")
    verified_incidents = relationship("IncidentReport", back_populates="verifier", foreign_keys="IncidentReport.verified_by")
    assigned_incidents = relationship("IncidentReport", back_populates="assignee", foreign_keys="IncidentReport.assigned_to")

class ChatHistory(Base):
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=False)
    role = Column(String(20), nullable=False)  # "user" or "assistant"
    message = Column(Text, nullable=False)
    chat_metadata = Column(JSON, nullable=True)  # Changed from 'metadata' to 'chat_metadata'
    created_at = Column(DateTime, default=datetime.utcnow)

# ========== INCIDENT REPORTING MODELS ==========

class IncidentReport(Base):
    __tablename__ = "incident_reports"
    
    id = Column(String(36), primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    description = Column(Text, nullable=False)
    
    # ML Predictions
    incident_type = Column(String(50))  # Accident, Fire, Medical, etc.
    severity = Column(String(20))  # Low, Medium, High, Critical
    priority = Column(Integer)  # 1-5
    ml_confidence = Column(Float)  # 0.0-1.0
    
    # Location
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    address = Column(String(500))
    barangay = Column(String(100))
    city = Column(String(100), default="Calapan")
    province = Column(String(100), default="Oriental Mindoro")
    
    # Contact Information
    contact_number = Column(String(20))
    emergency_contact = Column(String(100))
    
    # Media Files
    image_paths = Column(JSON)  # List of image paths
    video_paths = Column(JSON)  # List of video paths
    
    # ML Analysis Results
    text_analysis = Column(JSON)  # Text classification results
    image_analysis = Column(JSON, nullable=True)  # Image analysis results
    video_analysis = Column(JSON, nullable=True)  # Video analysis results
    keywords = Column(JSON)  # Extracted keywords
    
    # Status and Tracking
    status = Column(String(20), default="pending")  # pending, verified, in_progress, resolved
    verified_by = Column(Integer, ForeignKey("users.id"), nullable=True)  # Admin/Responder
    assigned_to = Column(Integer, ForeignKey("users.id"), nullable=True)  # Assigned responder
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    reporter = relationship("User", foreign_keys=[user_id], back_populates="reported_incidents")
    verifier = relationship("User", foreign_keys=[verified_by], back_populates="verified_incidents")
    assignee = relationship("User", foreign_keys=[assigned_to], back_populates="assigned_incidents")
    
    def to_dict(self):
        return {
            "id": self.id,
            "reporter": self.reporter.full_name if self.reporter else None,
            "description": self.description,
            "incident_type": self.incident_type,
            "severity": self.severity,
            "priority": self.priority,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "address": self.address,
            "barangay": self.barangay,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "ml_confidence": self.ml_confidence,
            "keywords": self.keywords or []
        }

class TrainingDataset(Base):
    __tablename__ = "training_datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(String(36), ForeignKey("incident_reports.id"))
    
    # Original Data
    description = Column(Text, nullable=False)
    incident_type = Column(String(50))
    severity = Column(String(20))
    
    # Corrected Labels (for supervised learning)
    corrected_type = Column(String(50))
    corrected_severity = Column(String(20))
    
    # Media References
    image_paths = Column(JSON)
    video_paths = Column(JSON)
    
    # Metadata
    is_verified = Column(Boolean, default=False)
    verified_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    verification_notes = Column(Text)
    
    # For model training
    used_in_training = Column(Boolean, default=False)
    training_version = Column(String(50), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    verified_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    report = relationship("IncidentReport")
    verifier = relationship("User")

class MLModelVersion(Base):
    __tablename__ = "ml_model_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    model_type = Column(String(50))  # text_classifier, image_classifier, severity_predictor
    version = Column(String(50))
    model_path = Column(String(500))
    
    # Performance Metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    training_loss = Column(Float)
    validation_loss = Column(Float)
    
    # Training Data Info
    training_samples = Column(Integer)
    validation_samples = Column(Integer)
    training_duration = Column(Float)  # in seconds
    
    # Status
    is_active = Column(Boolean, default=False)
    is_production = Column(Boolean, default=False)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    trained_by = Column(String(100))
    notes = Column(Text)

class IncidentCategory(Base):
    __tablename__ = "incident_categories"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True)
    description = Column(Text)
    severity_weight = Column(Float, default=1.0)  # Weight for severity calculation
    default_priority = Column(Integer, default=2)  # Default priority 1-5
    icon = Column(String(50))  # Icon for UI
    color = Column(String(20))  # Color code
    
    # Response guidelines
    response_time_target = Column(Integer)  # Target response time in minutes
    required_responders = Column(String(200))  # Comma-separated responder types
    
    def __repr__(self):
        return f"<IncidentCategory {self.name}>"

class BarangayInfo(Base):
    __tablename__ = "barangay_info"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True)
    population = Column(Integer)
    area_sqkm = Column(Float)
    priority_level = Column(Integer, default=2)  # 1-5 priority
    
    # Emergency Facilities
    has_hospital = Column(Boolean, default=False)
    has_fire_station = Column(Boolean, default=False)
    has_police_station = Column(Boolean, default=False)
    
    # Contact Information
    barangay_captain = Column(String(100))
    contact_number = Column(String(20))
    emergency_contact = Column(String(20))
    
    # Location
    latitude = Column(Float)
    longitude = Column(Float)
    
    def __repr__(self):
        return f"<BarangayInfo {self.name}>"

class EmergencyFacility(Base):
    __tablename__ = "emergency_facilities"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200))
    facility_type = Column(String(50))  # hospital, fire_station, police_station, clinic
    latitude = Column(Float)
    longitude = Column(Float)
    contact_number = Column(String(20))
    capacity = Column(Integer)
    is_available = Column(Boolean, default=True)
    
    # Operational hours
    open_24_7 = Column(Boolean, default=True)
    opening_time = Column(String(10), nullable=True)
    closing_time = Column(String(10), nullable=True)
    
    # Additional info
    barangay = Column(String(100))
    address = Column(String(500))
    notes = Column(Text)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class IncidentUpdate(Base):
    __tablename__ = "incident_updates"
    
    id = Column(Integer, primary_key=True, index=True)
    incident_id = Column(String(36), ForeignKey("incident_reports.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    
    update_type = Column(String(50))  # status_change, comment, location_update, media_added
    content = Column(Text)
    update_metadata = Column(JSON)  # Changed from 'metadata' to 'update_metadata'
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    incident = relationship("IncidentReport")
    user = relationship("User")


class AnonymousEmergency(Base):
    __tablename__ = "anonymous_emergencies"

    id = Column(Integer, primary_key=True, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    audio_path = Column(String(255), nullable=False)   # filesystem path
    timestamp = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String(45), nullable=True)    # optional client IP

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    message = Column(String(500), nullable=False)
    severity = Column(String(20), default="medium")   # low, medium, high, critical
    target_zone = Column(String(100), nullable=True)  # optional barangay or "all"
    target_roles = Column(JSON, default=[])           # list of roles to target
    geometry = Column(JSON, nullable=True)            # GeoJSON (LineString, Point, etc.)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)      # optional expiry
    image_url = Column(String(500), nullable=True)   # path to uploaded image

    # relationship
    creator = relationship("User", foreign_keys=[created_by])


class IncidentAssignmentLog(Base):
    __tablename__ = "incident_assignment_log"
    
    id = Column(Integer, primary_key=True, index=True)
    incident_id = Column(String(36), ForeignKey("incident_reports.id", ondelete="CASCADE"), nullable=False)
    assigned_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    assigned_to = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(SQLAlchemyEnum("assign", "unassign", name="assignment_action"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships (optional)
    incident = relationship("IncidentReport", backref="assignment_logs")
    assigner = relationship("User", foreign_keys=[assigned_by])
    assignee = relationship("User", foreign_keys=[assigned_to])

class ResponderLocation(Base):
    __tablename__ = "responder_locations"

    id = Column(Integer, primary_key=True, index=True)
    responder_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    accuracy = Column(Float, nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    responder = relationship("User", foreign_keys=[responder_id])

class LegalCompliance(Base):
    __tablename__ = "legal_compliances"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    law_number = Column(String, nullable=True)       # e.g., "RA 10121"
    category = Column(String, nullable=False)        # e.g., "Disaster Risk", "Traffic", "Emergency"
    description = Column(Text, nullable=False)
    official_statement = Column(Text, nullable=False)
    effective_date = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ResponderResolvedIncident(Base):
    __tablename__ = "responder_resolved_incidents"

    id = Column(Integer, primary_key=True, index=True)
    incident_id = Column(String(36), ForeignKey("incident_reports.id", ondelete="CASCADE"), nullable=False)
    responder_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    resolved_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)

    incident = relationship("IncidentReport", backref="resolved_logs")
    responder = relationship("User", backref="resolved_incidents")