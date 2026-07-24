# train_ml_final.py
"""
Final ML Training Script with correct parameter names
"""
import os
import sys
import json
import asyncio
from datetime import datetime

print("🚀 RESQAPP ML Training System (Final)")
print("=" * 60)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Ensure directories exist
os.makedirs("ml_models/saved_models/text_classifier", exist_ok=True)
os.makedirs("ml_models/saved_models/yolo", exist_ok=True)
os.makedirs("datasets", exist_ok=True)

def create_enhanced_classifier():
    """Create EnhancedIncidentClassifier inline to avoid import issues"""
    
    class EnhancedIncidentClassifier:
        def __init__(self, model_path="ml_models/saved_models/text_classifier/"):
            self.model_path = model_path
            self.labels = ["Accident", "Fire", "Medical", "Crime", "Natural Disaster", "Infrastructure", "Other"]
            
            os.makedirs(model_path, exist_ok=True)
            
            # Try to import ML libraries
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                self.ml_available = True
                print("✓ ML libraries available")
            except ImportError:
                self.ml_available = False
                print("⚠️ ML libraries not available. Using keyword-based classifier.")
            
            # Load model if exists
            self.load_model()
            
            # Emergency keywords
            self.keywords = {
                "Accident": ["accident", "crash", "collision", "vehicle", "car", "motorcycle", "truck", "hit", "wreck"],
                "Fire": ["fire", "burn", "flame", "smoke", "blaze", "arson"],
                "Medical": ["medical", "hospital", "doctor", "injured", "hurt", "pain", "unconscious", "bleeding"],
                "Crime": ["robbery", "theft", "assault", "shooting", "burglary", "crime", "attack"],
                "Natural Disaster": ["flood", "typhoon", "storm", "landslide", "earthquake", "disaster"],
                "Infrastructure": ["power", "outage", "water", "pipe", "road", "bridge", "telecom", "utility"],
                "Other": []
            }
        
        def load_model(self):
            """Try to load trained model"""
            try:
                if self.ml_available:
                    from transformers import AutoTokenizer, AutoModelForSequenceClassification
                    if os.path.exists(os.path.join(self.model_path, "config.json")):
                        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
                        print("✓ Loaded trained BERT model")
                        return True
            except Exception as e:
                print(f"⚠️ Could not load model: {e}")
            return False
        
        def predict(self, text):
            """Predict incident type"""
            if not text:
                return self.default_prediction()
            
            # Try BERT if available and loaded
            if hasattr(self, 'model') and self.model and self.ml_available:
                try:
                    return self.bert_predict(text)
                except Exception as e:
                    print(f"⚠️ BERT prediction failed: {e}")
                    # Fallback to keyword
                    pass
            
            # Fallback to keyword matching
            return self.keyword_predict(text)
        
        def bert_predict(self, text):
            """BERT-based prediction"""
            import torch
            
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                probs = probabilities.squeeze().tolist()
                predicted_idx = torch.argmax(probabilities, dim=-1).item()
            
            predicted_label = self.labels[predicted_idx] if predicted_idx < len(self.labels) else "Other"
            confidence = float(probs[predicted_idx])
            
            return {
                "type": predicted_label,
                "confidence": confidence,
                "all_predictions": probs,
                "model": "bert"
            }
        
        def keyword_predict(self, text):
            """Keyword-based prediction"""
            text_lower = text.lower()
            scores = {}
            
            for label, keywords in self.keywords.items():
                score = 0
                for keyword in keywords:
                    if keyword in text_lower:
                        score += 1
                scores[label] = score
            
            # Find best match
            max_score = max(scores.values())
            if max_score == 0:
                predicted_label = "Other"
                confidence = 0.5
            else:
                best_labels = [label for label, score in scores.items() if score == max_score]
                predicted_label = best_labels[0]
                confidence = min(0.3 + (max_score * 0.1), 0.8)
            
            # Create probability distribution
            total_score = sum(scores.values())
            if total_score > 0:
                probs = {label: (score / total_score) for label, score in scores.items()}
            else:
                probs = {label: 1/len(self.labels) for label in self.labels}
            
            return {
                "type": predicted_label,
                "confidence": confidence,
                "all_predictions": probs,
                "model": "keyword"
            }
        
        def default_prediction(self):
            return {
                "type": "Other",
                "confidence": 0.5,
                "all_predictions": {label: 1/len(self.labels) for label in self.labels},
                "model": "fallback"
            }
        
        def train_bert(self, epochs=3):
            """Train BERT model with correct parameter names"""
            if not self.ml_available:
                print("❌ ML libraries not available. Install with: pip install torch transformers datasets")
                return {"success": False, "error": "ML libraries not installed"}
            
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
                from datasets import Dataset
                from sklearn.metrics import accuracy_score, precision_recall_fscore_support
                import numpy as np
                
                print("📊 Generating training data...")
                training_data = self.generate_training_data()
                
                print(f"📊 Training on {len(training_data)} examples...")
                
                # Prepare dataset
                texts = [item["text"] for item in training_data]
                labels = [self.labels.index(item["label"]) for item in training_data]
                
                # Create dataset
                dataset = Dataset.from_dict({
                    "text": texts,
                    "label": labels
                })
                
                # Split
                split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
                train_dataset = split_dataset["train"]
                val_dataset = split_dataset["test"]
                
                # Tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                
                def tokenize_function(examples):
                    return self.tokenizer(
                        examples["text"],
                        padding="max_length",
                        truncation=True,
                        max_length=128
                    )
                
                tokenized_train = train_dataset.map(tokenize_function, batched=True)
                tokenized_val = val_dataset.map(tokenize_function, batched=True)
                
                # Model
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    "bert-base-uncased",
                    num_labels=len(self.labels)
                )
                
                # Check transformers version for correct parameter names
                import transformers
                transformers_version = transformers.__version__
                print(f"🤖 Using transformers v{transformers_version}")
                
                # Training args - compatible with different versions
                try:
                    # Try newer parameter names first
                    training_args = TrainingArguments(
                        output_dir=self.model_path,
                        num_train_epochs=epochs,
                        per_device_train_batch_size=8,
                        per_device_eval_batch_size=8,
                        warmup_steps=100,
                        weight_decay=0.01,
                        logging_dir=f"{self.model_path}/logs",
                        logging_steps=10,
                        eval_strategy="epoch",  # NEW parameter name
                        save_strategy="epoch",   # NEW parameter name
                        load_best_model_at_end=True,
                        metric_for_best_model="accuracy"
                    )
                    print("✓ Using new parameter names (eval_strategy, save_strategy)")
                except TypeError:
                    # Fallback to old parameter names
                    training_args = TrainingArguments(
                        output_dir=self.model_path,
                        num_train_epochs=epochs,
                        per_device_train_batch_size=8,
                        per_device_eval_batch_size=8,
                        warmup_steps=100,
                        weight_decay=0.01,
                        logging_dir=f"{self.model_path}/logs",
                        logging_steps=10,
                        evaluation_strategy="epoch",  # OLD parameter name
                        save_strategy="epoch",
                        load_best_model_at_end=True,
                        metric_for_best_model="accuracy"
                    )
                    print("✓ Using old parameter names (evaluation_strategy)")
                
                # Metrics
                def compute_metrics(pred):
                    labels = pred.label_ids
                    preds = pred.predictions.argmax(-1)
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        labels, preds, average='weighted'
                    )
                    acc = accuracy_score(labels, preds)
                    return {
                        "accuracy": acc,
                        "f1": f1,
                        "precision": precision,
                        "recall": recall
                    }
                
                # Trainer
                trainer = Trainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=tokenized_train,
                    eval_dataset=tokenized_val,
                    compute_metrics=compute_metrics
                )
                
                # Train
                print("🤖 Training BERT model (this may take a few minutes)...")
                trainer.train()
                
                # Save
                trainer.save_model(self.model_path)
                self.tokenizer.save_pretrained(self.model_path)
                
                # Evaluate
                eval_results = trainer.evaluate()
                
                # Save report
                report = {
                    "training_date": datetime.now().isoformat(),
                    "num_samples": len(training_data),
                    "results": eval_results,
                    "labels": self.labels,
                    "transformers_version": transformers_version
                }
                
                report_path = os.path.join(self.model_path, "training_report.json")
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2)
                
                # Test the model
                print(f"\n✅ Training complete! Accuracy: {eval_results.get('eval_accuracy', 0):.2%}")
                
                # Quick test
                print("\n🧪 Quick test after training:")
                test_texts = ["Car accident", "House fire", "Medical emergency"]
                for text in test_texts:
                    result = self.predict(text)
                    print(f"   '{text}' → {result['type']} ({result['confidence']:.1%})")
                
                return {
                    "success": True,
                    "accuracy": eval_results.get('eval_accuracy', 0),
                    "model_path": self.model_path,
                    "report_path": report_path
                }
                
            except Exception as e:
                print(f"❌ Training error: {e}")
                import traceback
                traceback.print_exc()
                
                # Try simplified training
                return self.train_simplified(training_data, epochs)
        
        def train_simplified(self, training_data, epochs=3):
            """Simplified training for compatibility"""
            try:
                print("🔄 Trying simplified training approach...")
                
                import torch
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                from torch.utils.data import DataLoader, Dataset as TorchDataset
                from torch.optim import AdamW
                import numpy as np
                
                # Custom PyTorch Dataset
                class IncidentDataset(TorchDataset):
                    def __init__(self, texts, labels, tokenizer, max_length=128):
                        self.texts = texts
                        self.labels = labels
                        self.tokenizer = tokenizer
                        self.max_length = max_length
                    
                    def __len__(self):
                        return len(self.texts)
                    
                    def __getitem__(self, idx):
                        text = str(self.texts[idx])
                        label = self.labels[idx]
                        
                        encoding = self.tokenizer(
                            text,
                            truncation=True,
                            padding='max_length',
                            max_length=self.max_length,
                            return_tensors='pt'
                        )
                        
                        return {
                            'input_ids': encoding['input_ids'].flatten(),
                            'attention_mask': encoding['attention_mask'].flatten(),
                            'labels': torch.tensor(label, dtype=torch.long)
                        }
                
                # Prepare data
                texts = [item["text"] for item in training_data]
                labels = [self.labels.index(item["label"]) for item in training_data]
                
                # Split
                split_idx = int(0.8 * len(texts))
                train_texts, val_texts = texts[:split_idx], texts[split_idx:]
                train_labels, val_labels = labels[:split_idx], labels[split_idx:]
                
                # Tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                
                # Datasets
                train_dataset = IncidentDataset(train_texts, train_labels, self.tokenizer)
                val_dataset = IncidentDataset(val_texts, val_labels, self.tokenizer)
                
                # DataLoaders
                train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=8)
                
                # Model
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    "bert-base-uncased",
                    num_labels=len(self.labels)
                )
                
                # Training
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model.to(device)
                
                optimizer = AdamW(self.model.parameters(), lr=2e-5)
                
                print(f"📊 Training on {len(train_dataset)} examples...")
                
                for epoch in range(epochs):
                    # Training
                    self.model.train()
                    total_loss = 0
                    
                    for batch in train_loader:
                        optimizer.zero_grad()
                        
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)
                        
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        
                        loss = outputs.loss
                        total_loss += loss.item()
                        
                        loss.backward()
                        optimizer.step()
                    
                    # Validation
                    self.model.eval()
                    correct = 0
                    total = 0
                    
                    with torch.no_grad():
                        for batch in val_loader:
                            input_ids = batch['input_ids'].to(device)
                            attention_mask = batch['attention_mask'].to(device)
                            labels = batch['labels'].to(device)
                            
                            outputs = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask
                            )
                            
                            predictions = torch.argmax(outputs.logits, dim=-1)
                            correct += (predictions == labels).sum().item()
                            total += labels.size(0)
                    
                    accuracy = correct / total if total > 0 else 0
                    avg_loss = total_loss / len(train_loader)
                    
                    print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")
                
                # Save model
                self.model.save_pretrained(self.model_path)
                self.tokenizer.save_pretrained(self.model_path)
                
                # Save report
                report = {
                    "training_date": datetime.now().isoformat(),
                    "num_samples": len(training_data),
                    "accuracy": accuracy,
                    "labels": self.labels,
                    "method": "simplified_training"
                }
                
                report_path = os.path.join(self.model_path, "training_report.json")
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2)
                
                print(f"✅ Simplified training complete! Final accuracy: {accuracy:.2%}")
                
                self.use_bert = True
                
                return {
                    "success": True,
                    "accuracy": accuracy,
                    "model_path": self.model_path,
                    "report_path": report_path
                }
                
            except Exception as e:
                print(f"❌ Simplified training also failed: {e}")
                return {"success": False, "error": str(e)}
        
        def generate_training_data(self):
            """Generate synthetic training data"""
            data = []
            
            templates = {
                "Accident": [
                    "Car accident on Main Road with injuries",
                    "Vehicle collision at intersection",
                    "Motorcycle crash near market",
                    "Truck accident on highway",
                    "Two cars collided on national road",
                    "Pedestrian hit by vehicle",
                    "Bus accident with passengers",
                    "Traffic accident causing road blockage"
                ],
                "Fire": [
                    "Fire in residential building, smoke visible",
                    "House fire on Rizal Street",
                    "Building fire with flames",
                    "Vehicle fire on highway",
                    "Electrical fire in commercial building",
                    "Forest fire spreading quickly",
                    "Kitchen fire in apartment",
                    "Warehouse fire with toxic smoke"
                ],
                "Medical": [
                    "Medical emergency, person unconscious",
                    "Heart attack at shopping mall",
                    "Person injured in accident",
                    "Breathing difficulty emergency",
                    "Stroke patient needs ambulance",
                    "Child with high fever and seizures",
                    "Elderly person collapsed in park",
                    "Worker with serious injury"
                ],
                "Crime": [
                    "Robbery at convenience store",
                    "Theft reported in Tawagan area",
                    "Assault on Burgos Avenue",
                    "Shooting incident in Poblacion",
                    "Burglary at residential house",
                    "Car theft in parking lot",
                    "Vandalism at public park",
                    "Drug-related incident reported"
                ],
                "Natural Disaster": [
                    "Flooding in Tawagan due to heavy rain",
                    "Typhoon damage in coastal areas",
                    "Landslide on mountain road",
                    "Earthquake felt in Calapan City",
                    "Storm damage to houses",
                    "Flash flood in low-lying area",
                    "Severe weather causing outages",
                    "River overflow affecting homes"
                ],
                "Infrastructure": [
                    "Power outage affecting entire barangay",
                    "Water pipe burst on M.H. Del Pilar",
                    "Road damage causing traffic issues",
                    "Bridge collapse in rural area",
                    "Telecommunication lines down",
                    "Sewage system overflow",
                    "Gas leak in commercial district",
                    "Building structure damage"
                ],
                "Other": [
                    "Strange incident in Central area",
                    "Emergency situation needs response",
                    "Help needed for unknown situation",
                    "Public disturbance reported",
                    "Animal threat in neighborhood",
                    "Lost person in forest area",
                    "Suspicious package found",
                    "General emergency request"
                ]
            }
            
            for label, examples in templates.items():
                for example in examples:
                    variations = [
                        f"URGENT: {example}",
                        f"{example}. Please send help.",
                        f"EMERGENCY: {example}",
                        f"REPORT: {example}",
                        example,
                        example.lower(),
                    ]
                    
                    for variation in variations[:3]:  # Take 3 variations
                        data.append({
                            "text": variation,
                            "label": label
                        })
            
            print(f"✅ Generated {len(data)} training examples")
            return data
    
    return EnhancedIncidentClassifier

async def train_text_classifier():
    """Train text classifier"""
    print("\n📊 Training Text Classifier")
    print("=" * 40)
    
    EnhancedIncidentClassifier = create_enhanced_classifier()
    classifier = EnhancedIncidentClassifier()
    
    result = classifier.train_bert(epochs=3)
    
    if result.get("success", False):
        print(f"\n✅ Training successful!")
        print(f"   Model saved to: {result['model_path']}")
        print(f"   Accuracy: {result.get('accuracy', 0):.2%}")
        
        # More comprehensive test
        print("\n🧪 Comprehensive test:")
        test_cases = [
            "Car accident on Main Road",
            "Fire in building with smoke",
            "Medical emergency at hospital",
            "Flooding in Tawagan area",
            "Power outage in barangay",
            "Robbery at convenience store",
            "General emergency help needed"
        ]
        
        for text in test_cases:
            prediction = classifier.predict(text)
            print(f"   '{text}'")
            print(f"     → Type: {prediction['type']}")
            print(f"     → Confidence: {prediction['confidence']:.1%}")
            print(f"     → Model: {prediction['model']}")
        
        return True
    else:
        print(f"\n❌ Training failed: {result.get('error', 'Unknown error')}")
        return False

async def check_dependencies():
    """Check and install dependencies"""
    print("\n🔍 Checking dependencies...")
    
    dependencies = [
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("datasets", "HuggingFace Datasets"),
        ("scikit-learn", "Scikit-learn"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
    ]
    
    all_ok = True
    
    for package, name in dependencies:
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name}")
            all_ok = False
    
    if not all_ok:
        print(f"\n⚠️ Some packages are missing.")
        print(f"   Recommended install: pip install torch transformers datasets scikit-learn pandas")
    
    return all_ok

async def check_transformers_version():
    """Check transformers version"""
    try:
        import transformers
        version = transformers.__version__
        print(f"\n🤖 Transformers version: {version}")
        
        # Check if version is compatible
        from packaging import version as pkg_version
        current = pkg_version.parse(version)
        if current >= pkg_version.parse("4.0.0"):
            print("✓ Compatible version (uses eval_strategy)")
        else:
            print("⚠ Older version (uses evaluation_strategy)")
        
        return True
    except:
        print("⚠ Could not check transformers version")
        return False

async def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("🎯 RESQAPP ML TRAINING MENU")
    print("=" * 60)
    print("1. Train Text Classifier (BERT)")
    print("2. Check Dependencies")
    print("3. Check Transformers Version")
    print("4. Test Existing Models")
    print("5. Exit")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        await check_transformers_version()
        deps_ok = await check_dependencies()
        if deps_ok or input("\nContinue anyway? (y/n): ").lower() == 'y':
            await train_text_classifier()
    
    elif choice == "2":
        await check_dependencies()
    
    elif choice == "3":
        await check_transformers_version()
    
    elif choice == "4":
        print("\n🧪 Testing existing models...")
        EnhancedIncidentClassifier = create_enhanced_classifier()
        classifier = EnhancedIncidentClassifier()
        
        # Check if model exists
        model_exists = os.path.exists(os.path.join(classifier.model_path, "config.json"))
        if model_exists:
            print("✓ Found trained model")
        else:
            print("⚠ No trained model found, using keyword-based classifier")
        
        # Test predictions
        test_texts = [
            "Car accident with injuries on highway",
            "House fire with smoke visible",
            "Medical emergency at hospital",
            "Power outage in entire barangay",
            "Flooding in Tawagan due to heavy rain",
            "Robbery at 24/7 store",
            "General help needed"
        ]
        
        for text in test_texts:
            result = classifier.predict(text)
            print(f"\n📝: {text}")
            print(f"   Type: {result['type']}")
            print(f"   Confidence: {result['confidence']:.1%}")
            print(f"   Model: {result['model']}")
    
    elif choice == "5":
        print("👋 Exiting...")
        return
    
    else:
        print("❌ Invalid choice")
    
    print("\n" + "=" * 60)
    print("🎉 ML Training Complete!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())