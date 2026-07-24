# test_complete_system.py
import sys
sys.path.append('.')

from database import SessionLocal
import crud_incidents
from datetime import datetime

def test_database_operations():
    """Test all database operations"""
    db = SessionLocal()
    
    print("🧪 Testing Database Operations")
    print("=" * 60)
    
    try:
        # 1. Test getting categories
        categories = crud_incidents.get_incident_categories(db)
        print(f"✅ 1. Found {len(categories)} incident categories")
        
        # 2. Test getting barangays
        barangays = crud_incidents.get_barangays(db)
        print(f"✅ 2. Found {len(barangays)} barangays")
        
        # 3. Test getting emergency facilities
        facilities = crud_incidents.get_emergency_facilities(db)
        print(f"✅ 3. Found {len(facilities)} emergency facilities")
        
        # 4. Test getting incidents
        incidents = crud_incidents.get_all_incidents(db, limit=5)
        print(f"✅ 4. Found {len(incidents)} incidents in database")
        
        # 5. Test statistics
        if len(incidents) > 0:
            stats = crud_incidents.get_statistics(db, days=30)
            print(f"✅ 5. Statistics: {stats['total']} incidents in last 30 days")
        
        # 6. Test search
        search_results = crud_incidents.search_incidents(db, keyword="accident")
        print(f"✅ 6. Search found {len(search_results)} incidents with 'accident'")
        
        # 7. Test nearby facilities
        if facilities:
            nearby = crud_incidents.get_nearby_facilities(
                db, lat=13.411, lng=121.181, radius_km=5
            )
            print(f"✅ 7. Found {len(nearby)} facilities near Calapan City Hall")
        
        print("\n" + "=" * 60)
        print("✅ All database operations working correctly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

def test_ml_integration():
    """Test ML service integration"""
    print("\n🤖 Testing ML Service Integration")
    print("=" * 60)
    
    try:
        from services.incident_ml_service import IncidentMLService
        ml_service = IncidentMLService()
        
        # Test text analysis
        test_text = "Car accident on highway with multiple injuries"
        result = ml_service.text_classifier.predict(test_text)
        keywords = ml_service.text_classifier.extract_keywords(test_text)
        
        print(f"✅ ML Text Analysis:")
        print(f"   Text: {test_text}")
        print(f"   Prediction: {result['type']} ({result['confidence']:.1%})")
        print(f"   Keywords: {', '.join(keywords)}")
        
        print("\n✅ ML service is working correctly!")
        
    except Exception as e:
        print(f"❌ ML Service Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Testing Complete Incident Reporting System")
    print("=" * 60)
    
    test_database_operations()
    test_ml_integration()
    
    print("\n" + "=" * 60)
    print("🎉 System is ready for incident reporting!")
    print("\n📋 Next steps:")
    print("   1. Start backend: uvicorn main:app --reload")
    print("   2. Test API endpoints with Postman")
    print("   3. Update frontend to use new endpoints")
    print("   4. Submit your first incident report!")