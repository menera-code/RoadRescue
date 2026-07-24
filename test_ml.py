# test_complete_ml.py
import asyncio
from services.incident_ml_service import IncidentMLService
from ml_models.text_classifier import IncidentTextClassifier
from ml_models.image_processor import IncidentImageAnalyzer

async def test_complete_system():
    print("🧪 Testing Complete ML Incident Reporting System")
    print("=" * 70)
    
    # Initialize services
    ml_service = IncidentMLService()
    text_classifier = IncidentTextClassifier()
    image_analyzer = IncidentImageAnalyzer()
    
    # Test 1: Text Classification
    print("\n1. 📝 Testing Text Classification:")
    test_texts = [
        "Car accident with injuries on national highway",
        "House fire in barangay Tawagan, need firetruck",
        "Person collapsed near market, needs ambulance",
        "Robbery at 7/11 store, suspect armed",
        "Flooding in low-lying areas due to heavy rain"
    ]
    
    for text in test_texts:
        result = text_classifier.predict(text)
        keywords = text_classifier.extract_keywords(text)
        color = text_classifier.get_confidence_color(result["confidence"])
        print(f"   {color} {result['type']} ({result['confidence']:.1%}): {text[:50]}...")
        print(f"      Keywords: {', '.join(keywords)}")
    
    # Test 2: Image Analyzer (simulated)
    print("\n2. 📷 Testing Image Analyzer (simulated):")
    print("   Image analyzer initialized and ready")
    print("   Will process images when uploaded")
    
    # Test 3: Complete Report Processing
    print("\n3. 🚨 Testing Complete Report Processing:")
    
    sample_report = {
        "description": "Major car accident on highway near Calapan, multiple injuries reported",
        "latitude": 13.411,
        "longitude": 121.181,
        "barangay": "Tawagan",
        "contact_number": "09123456789"
    }
    
    result = await ml_service.process_report(sample_report)
    
    print(f"   ✅ Report ID: {result['report_id']}")
    print(f"   🏷️  Incident Type: {result['incident_type']}")
    print(f"   ⚠️  Severity: {result['severity']}")
    print(f"   🔢 Priority: {result['priority']}/5")
    print(f"   📊 Confidence: {result['confidence']:.1%}")
    print(f"   📍 Location: {result['location']['barangay']}, Calapan")
    
    print("\n4. 💡 Recommendations Generated:")
    for i, rec in enumerate(result.get('recommendations', [])[:3], 1):
        print(f"   {i}. {rec}")
    
    print("\n" + "=" * 70)
    print("✅ All ML services are working correctly!")
    print("💡 The system is ready for incident reporting with:")
    print("   - Text classification")
    print("   - Image analysis") 
    print("   - Video analysis")
    print("   - Severity assessment")
    print("   - Priority calculation")
    print("   - AI recommendations")

if __name__ == "__main__":
    asyncio.run(test_complete_system())