# backend/ml_train/train_image.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from database import SessionLocal
from ml_train.data_loader import TrainingDataLoader
from models import MLModelVersion
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def train_image_model():
    print("Loading image data...")
    db = SessionLocal()
    loader = TrainingDataLoader(db)
    images, labels_raw = loader.load_image_data()
    db.close()
    if not images:
        print("No image data found.")
        return

    # Encode labels
    le = LabelEncoder()
    labels = le.fit_transform(labels_raw)
    num_classes = len(le.classes_)
    print(f"Classes: {le.classes_}")
    print(f"Total images: {len(images)}")

    # Split
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageDataset(X_train, y_train, train_transform)
    val_dataset = ImageDataset(X_val, y_val, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Load pretrained EfficientNet
    model = models.efficientnet_b0(pretrained=True)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    best_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        print(f"Validation Accuracy: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_model_state = model.state_dict()
            os.makedirs('./models/image', exist_ok=True)
            torch.save(best_model_state, './models/image/efficientnet_final.pth')
            joblib.dump(le, './models/image/label_encoder.pkl')
            print(f"✅ Saved best model with accuracy {best_acc:.2f}%")

    # ------------------------------------------------------------------
    # AFTER TRAINING: Record model version in database
    # ------------------------------------------------------------------
    # best_acc is the percentage, but database expects a float (0.0-1.0)
    best_accuracy = best_acc / 100.0

    db = SessionLocal()
    try:
        # Count existing versions
        version_count = db.query(MLModelVersion).filter(
            MLModelVersion.model_type == "image_classifier"
        ).count()
        version_num = version_count + 1

        version = MLModelVersion(
            model_type="image_classifier",
            version=f"v{version_num}",
            model_path="./models/image/efficientnet_final.pth",
            accuracy=best_accuracy,
            training_samples=len(X_train),
            validation_samples=len(X_val),
            is_active=True,
            is_production=True,
            trained_by="admin",
            notes="Fine-tuned EfficientNet on incident images"
        )
        db.add(version)
        db.commit()
        print(f"✅ Model version {version_num} recorded in database with accuracy {best_accuracy:.4f}")
    except Exception as e:
        print(f"❌ Failed to save model version: {e}")
    finally:
        db.close()

    print("Image model training completed.")

if __name__ == "__main__":
    train_image_model()