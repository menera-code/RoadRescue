import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.enhanced_incident_ml_service import EnhancedIncidentMLService
import asyncio
import json
from datetime import datetime

async def main():
    print("🤖 RESQAPP ML Training System")
    print("=" * 50)
    
    # Initialize enhanced ML service
    ml_service = EnhancedIncidentMLService(use_enhanced=True)
    
    # Check model status
    print("\n📊 Model Status:")
    print(f"• Text Classifier: {'Enhanced' if ml_service.use_enhanced else 'Basic'}")
    print(f"• Image Analyzer: {'Enhanced' if ml_service.use_enhanced else 'Basic'}")
    print(f"• Video Analyzer: {'Enhanced' if ml_service.use_enhanced else 'Basic'}")
    
    # Train models
    train_choice = input("\n🚀 Train models? (y/n): ")
    if train_choice.lower() == 'y':
        print("\n🎯 Starting training process...")
        
        # Train text classifier
        print("\n1. Training Text Classifier (BERT)...")
        await ml_service.train_models()
        
        # Check for image dataset
        image_dataset = "datasets/incident_images"  # Update with your dataset path
        if os.path.exists(image_dataset):
            print("\n2. Training Image Analyzer (YOLO)...")
            # Uncomment when you have dataset
            # ml_service.image_analyzer.train_custom_model(image_dataset)
            print("⚠ Image dataset found but training requires manual setup")
        else:
            print("⚠ Image dataset not found at:", image_dataset)
        
        print("\n✅ Training complete!")
    
    # Test with sample data
    test_choice = input("\n🧪 Test with sample incident? (y/n): ")
    if test_choice.lower() == 'y':
        print("\n📝 Testing ML pipeline...")
        
        sample_report = {
            "description": "Car accident on main road with multiple injuries, urgent help needed",
            "latitude": 13.414,
            "longitude": 121.180,
            "barangay": "Tawagan",
            "address": "Main Road near Calapan City Hall"
        }
        
        result = await ml_service.process_report(sample_report)
        
        print("\n📊 Analysis Results:")
        print(f"• Incident Type: {result['incident_type']}")
        print(f"• Severity: {result['severity']}")
        print(f"• Priority: {result['priority']}")
        print(f"• Confidence: {result['confidence']:.1%}")
        print(f"• Model Type: {result['model_type']}")
        
        print("\n🎯 Recommendations:")
        for i, rec in enumerate(result['recommendations'][:3], 1):
            print(f"  {i}. {rec}")
    
    print("\n🎉 ML System Ready!")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())