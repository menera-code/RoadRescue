import os
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from database import SessionLocal
from models import IncidentReport, TrainingDataset, MLModelVersion
from services.enhanced_incident_ml_service import EnhancedIncidentMLService
from services.predictor import predict_text


class MLAnalyticsService:
    def __init__(self):
        self.ml_service = EnhancedIncidentMLService(use_enhanced=True)

    # ----------------------------------------------------------------------
    # USER-LEVEL STATISTICS
    # ----------------------------------------------------------------------
    def get_user_ml_stats(self, user_id: int, days: int = 30, db: Session = None) -> Dict[str, Any]:
        """Get ML prediction statistics for a specific user."""
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True

        try:
            cutoff = datetime.utcnow() - timedelta(days=days)

            incidents = db.query(
                IncidentReport.incident_type,
                IncidentReport.severity,
                IncidentReport.ml_confidence,
                IncidentReport.created_at
            ).filter(
                IncidentReport.user_id == user_id,
                IncidentReport.created_at >= cutoff,
                IncidentReport.ml_confidence.isnot(None)
            ).all()

            stats = {
                "total_predictions": len(incidents),
                "by_type": {},
                "by_severity": {},
                "avg_confidence": 0.0,
                "trend_data": [],
                "days_analyzed": days,
                "top_predictions": {}
            }

            confidences = []
            weekly_stats = {}

            for inc in incidents:
                inc_type = inc.incident_type or "Unknown"
                stats["by_type"][inc_type] = stats["by_type"].get(inc_type, 0) + 1

                severity = inc.severity or "Unknown"
                stats["by_severity"][severity] = stats["by_severity"].get(severity, 0) + 1

                if inc.ml_confidence is not None:
                    confidences.append(inc.ml_confidence)

                week = inc.created_at.strftime("%Y-W%U")
                weekly_stats[week] = weekly_stats.get(week, 0) + 1

            if confidences:
                stats["avg_confidence"] = sum(confidences) / len(confidences)

            stats["trend_data"] = sorted(weekly_stats.items(), key=lambda x: x[0])

            stats["top_predictions"] = dict(
                sorted(stats["by_type"].items(), key=lambda x: x[1], reverse=True)[:5]
            )

            return stats

        finally:
            if close_db:
                db.close()

    # ----------------------------------------------------------------------
    # MODEL PERFORMANCE FROM DATABASE
    # ----------------------------------------------------------------------
    def get_model_performance(self, db: Session = None) -> Dict[str, Any]:
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True
        try:
            subquery = db.query(
                MLModelVersion.model_type,
                func.max(MLModelVersion.created_at).label("latest_created")
            ).filter(MLModelVersion.is_production == True).group_by(MLModelVersion.model_type).subquery()

            latest_models = db.query(MLModelVersion).join(
                subquery,
                and_(
                    MLModelVersion.model_type == subquery.c.model_type,
                    MLModelVersion.created_at == subquery.c.latest_created
                )
            ).all()

            result = {}
            for model in latest_models:
                result[model.model_type] = {
                    "version": model.version,
                    "accuracy": model.accuracy or 0.0,
                    "precision": model.precision or 0.0,
                    "recall": model.recall or 0.0,
                    "f1_score": model.f1_score or 0.0,
                    "training_samples": model.training_samples or 0,
                    "validation_samples": model.validation_samples or 0,
                    "trained_at": model.created_at.isoformat() if model.created_at else None
                }

            all_models = db.query(MLModelVersion).count()
            result["_meta"] = {
                "total_training_runs": all_models,
                "buffer_size": len(self.ml_service.training_buffer)
            }

            return result
        finally:
            if close_db:
                db.close()

    # ----------------------------------------------------------------------
    # DATASET STORAGE STATUS
    # ----------------------------------------------------------------------
    def get_dataset_status(self, db: Session = None) -> Dict[str, Any]:
        """Return storage info for image/video directories AND training dataset counts."""
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True

        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            incident_images = os.path.join(base_dir, "datasets/incident_images")
            training_data_dir = os.path.join(base_dir, "training_data")
            uploads_dir = os.path.join(base_dir, "uploads")

            status = {
                "incident_images": self._dir_info(incident_images),
                "training_data": self._dir_info(training_data_dir),
                "uploads": self._dir_info(uploads_dir),
                "total_size_mb": 0,
                "storage_warning": False
            }

            verified = db.query(TrainingDataset).filter(TrainingDataset.is_verified == True).count()
            used = db.query(TrainingDataset).filter(TrainingDataset.used_in_training == True).count()
            status["verified_samples"] = verified
            status["used_in_training"] = used

            total_mb = (status["incident_images"]["size_mb"] +
                        status["training_data"]["size_mb"] +
                        status["uploads"]["size_mb"])
            status["total_size_mb"] = total_mb
            if total_mb > 8000:
                status["storage_warning"] = True

            return status

        finally:
            if close_db:
                db.close()

    # ----------------------------------------------------------------------
    # TRAINING DATA STATUS
    # ----------------------------------------------------------------------
    def get_training_data_status(self, db: Session = None) -> Dict[str, Any]:
        """Return counts of verified and used training samples."""
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True

        try:
            verified = db.query(TrainingDataset).filter(TrainingDataset.is_verified == True).count()
            used = db.query(TrainingDataset).filter(TrainingDataset.used_in_training == True).count()
            return {
                "verified_samples": verified,
                "used_in_training": used
            }
        finally:
            if close_db:
                db.close()

    # ----------------------------------------------------------------------
    # SUMMARY STATISTICS OVER TIME
    # ----------------------------------------------------------------------
    def get_training_stats(self, days: int = 30, db: Session = None) -> Dict[str, Any]:
        """Return summary stats for recent predictions (by type, severity, confidence)."""
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True

        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            incidents = db.query(
                IncidentReport.incident_type,
                IncidentReport.severity,
                IncidentReport.ml_confidence
            ).filter(
                IncidentReport.created_at >= cutoff,
                IncidentReport.ml_confidence.isnot(None)
            ).all()

            total = len(incidents)
            by_type = {}
            by_severity = {}
            confidences = []

            for inc in incidents:
                t = inc.incident_type or "Unknown"
                by_type[t] = by_type.get(t, 0) + 1

                s = inc.severity or "Unknown"
                by_severity[s] = by_severity.get(s, 0) + 1

                if inc.ml_confidence:
                    confidences.append(inc.ml_confidence)

            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return {
                "total_predictions": total,
                "avg_confidence": avg_confidence,
                "by_type": by_type,
                "by_severity": by_severity,
                "days_analyzed": days
            }

        finally:
            if close_db:
                db.close()

    # ----------------------------------------------------------------------
    # HELPER: directory size info
    # ----------------------------------------------------------------------
    def _dir_info(self, path: str) -> Dict[str, Any]:
        """Return existence, file count, and size in MB for a directory."""
        info = {
            "exists": os.path.exists(path),
            "file_count": 0,
            "size_mb": 0.0
        }
        if os.path.exists(path) and os.path.isdir(path):
            try:
                files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
                info["file_count"] = len(files)
                total = 0
                for f in files:
                    total += os.path.getsize(os.path.join(path, f))
                info["size_mb"] = round(total / (1024 * 1024), 1)
            except Exception:
                pass
        return info

    # ----------------------------------------------------------------------
    # SYNTHETIC DATA GENERATION (FIXED - uses predictor.py)
    # ----------------------------------------------------------------------
    def generate_synthetic_sample(self, count: int = 100) -> Dict[str, Any]:
        """Generate small synthetic dataset for testing (storage safe)."""
        synthetic = []

        templates = {
            "Accident": ["Car crash on highway", "Motorcycle accident", "Pedestrian collision"],
            "Fire": ["House fire", "Vehicle fire", "Building blaze"],
            "Medical": ["Heart attack", "Unconscious person", "Injury from fall"]
        }

        for _ in range(count):
            inc_type = random.choice(list(templates.keys()))
            text = random.choice(templates[inc_type])
            # Use predictor.py instead of classifier
            prediction = predict_text(text)

            synthetic.append({
                "text": text,
                "predicted_type": prediction.get("incident_type", "Accident"),
                "confidence": prediction.get("type_confidence", 0.5),
                "true_label": inc_type
            })

        return {
            "success": True,
            "samples": synthetic,
            "size_bytes": len(json.dumps(synthetic).encode()),
            "storage_mb": round(len(json.dumps(synthetic).encode()) / (1024 * 1024), 2)
        }


# Global instance
ml_analytics = MLAnalyticsService()