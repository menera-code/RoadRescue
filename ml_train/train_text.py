# backend/ml_train/train_text.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from database import SessionLocal
from ml_train.data_loader import TrainingDataLoader
from models import MLModelVersion
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_text_model():
    print("Loading training data...")
    db = SessionLocal()
    loader = TrainingDataLoader(db)
    texts, labels_raw = loader.load_text_data()
    db.close()
    if not texts:
        print("No text data found.")
        return

    # Encode labels
    le = LabelEncoder()
    labels = le.fit_transform(labels_raw)
    num_classes = len(le.classes_)
    print(f"Classes: {le.classes_}")
    print(f"Total samples: {len(texts)}")

    # Split
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Tokenizer and model
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

    train_dataset = TextDataset(X_train, y_train, tokenizer)
    val_dataset = TextDataset(X_val, y_val, tokenizer)

    training_args = TrainingArguments(
        output_dir='./models/text',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        logging_steps=10,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    # Save model and label encoder
    os.makedirs('./models/text', exist_ok=True)
    model.save_pretrained('./models/text/bert_final')
    tokenizer.save_pretrained('./models/text/bert_final')
    joblib.dump(le, './models/text/label_encoder.pkl')
    print("✅ Text model saved to ./models/text/bert_final")

    # ------------------------------------------------------------------
    # AFTER TRAINING: Record model version in database
    # ------------------------------------------------------------------
    # Get the best validation accuracy from the trainer
    best_metric = trainer.state.best_metric
    # The metric name may be 'eval_accuracy' (depending on trainer)
    # We'll assume it's stored; if not, compute manually
    eval_accuracy = best_metric if best_metric else 0.0

    # If you want to compute more metrics, you can evaluate the best model
    # on the validation set manually. For simplicity, we use the best_metric.

    db = SessionLocal()
    try:
        # Count existing versions to generate version number
        version_count = db.query(MLModelVersion).filter(
            MLModelVersion.model_type == "text_classifier"
        ).count()
        version_num = version_count + 1

        version = MLModelVersion(
            model_type="text_classifier",
            version=f"v{version_num}",
            model_path="./models/text/bert_final",
            accuracy=eval_accuracy,
            training_samples=len(X_train),
            validation_samples=len(X_val),
            is_active=True,
            is_production=True,
            trained_by="admin",
            notes="Fine-tuned on incident reports"
        )
        db.add(version)
        db.commit()
        print(f"✅ Model version {version_num} recorded in database with accuracy {eval_accuracy:.4f}")
    except Exception as e:
        print(f"❌ Failed to save model version: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    train_text_model()