# backend/ml_train/data_loader.py
import json
import os
from sqlalchemy.orm import Session
from database import SessionLocal
from models import IncidentReport, TrainingDataset
from PIL import Image

class TrainingDataLoader:
    def __init__(self, db: Session):
        self.db = db

    def load_text_data(self):
        """Return (texts, labels) from verified training datasets."""
        training_records = self.db.query(TrainingDataset).filter(
            TrainingDataset.is_verified == True,
            TrainingDataset.corrected_type.isnot(None)
        ).all()

        texts, labels = [], []
        for record in training_records:
            incident = self.db.query(IncidentReport).filter(
                IncidentReport.id == record.report_id
            ).first()
            if incident and incident.description:
                texts.append(incident.description)
                labels.append(record.corrected_type)

        # Filter out labels that appear too few times (optional)
        from collections import Counter
        label_counts = Counter(labels)
        min_samples = 2
        valid_labels = {lbl for lbl, cnt in label_counts.items() if cnt >= min_samples}
        if len(valid_labels) < 2:
            print("Not enough data for training. Need at least 2 classes with min samples.")
            return None, None

        filtered_texts, filtered_labels = [], []
        for t, l in zip(texts, labels):
            if l in valid_labels:
                filtered_texts.append(t)
                filtered_labels.append(l)
        return filtered_texts, filtered_labels

    def load_image_data(self):
        """Return (images, labels) from verified reports that have images."""
        training_records = self.db.query(TrainingDataset).filter(
            TrainingDataset.is_verified == True,
            TrainingDataset.corrected_type.isnot(None)
        ).all()

        images, labels = [], []
        for record in training_records:
            incident = self.db.query(IncidentReport).filter(
                IncidentReport.id == record.report_id
            ).first()
            if incident and incident.image_paths:
                paths = json.loads(incident.image_paths) if isinstance(incident.image_paths, str) else incident.image_paths
                for path in paths:
                    full_path = os.path.join('uploads', path)
                    if os.path.exists(full_path):
                        try:
                            img = Image.open(full_path).convert('RGB')
                            images.append(img)
                            labels.append(record.corrected_type)
                        except:
                            continue

        if len(images) < 10:
            print(f"Only {len(images)} images, need at least 10.")
            return None, None
        return images, labels