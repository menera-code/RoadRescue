# migrate_incident_tables.py
import sys
sys.path.append('.')

from database import engine, SessionLocal
import models

def migrate_incident_tables():
    """Create incident-related tables in the database"""
    print("🔄 Creating incident reporting tables...")
    
    try:
        # Create all tables
        models.Base.metadata.create_all(bind=engine)
        print("✅ Incident tables created successfully!")
        
        # Add default data
        add_default_data()
        
        print("\n✅ Migration complete!")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()

def add_default_data():
    """Add default categories and barangays"""
    db = SessionLocal()
    
    try:
        # Add default incident categories
        categories = [
            {
                "name": "Accident",
                "description": "Road accidents, vehicle collisions, traffic incidents",
                "severity_weight": 1.2,
                "default_priority": 3,
                "icon": "🚗",
                "color": "#f59e0b",
                "response_time_target": 10,
                "required_responders": "police,ambulance"
            },
            {
                "name": "Fire",
                "description": "Fire incidents, structural fires, wildfires",
                "severity_weight": 1.5,
                "default_priority": 4,
                "icon": "🔥",
                "color": "#dc2626",
                "response_time_target": 5,
                "required_responders": "fire_truck,ambulance,police"
            },
            {
                "name": "Medical",
                "description": "Medical emergencies, injuries, health crises",
                "severity_weight": 1.4,
                "default_priority": 4,
                "icon": "🏥",
                "color": "#ef4444",
                "response_time_target": 8,
                "required_responders": "ambulance,paramedic"
            },
            {
                "name": "Crime",
                "description": "Criminal activities, theft, assault, robbery",
                "severity_weight": 1.3,
                "default_priority": 3,
                "icon": "🚔",
                "color": "#374151",
                "response_time_target": 10,
                "required_responders": "police"
            },
            {
                "name": "Natural Disaster",
                "description": "Floods, earthquakes, landslides, typhoons",
                "severity_weight": 1.6,
                "default_priority": 5,
                "icon": "🌊",
                "color": "#1d4ed8",
                "response_time_target": 15,
                "required_responders": "rescue_team,ambulance,police"
            },
            {
                "name": "Infrastructure",
                "description": "Road damage, power outages, water leaks, building issues",
                "severity_weight": 1.0,
                "default_priority": 2,
                "icon": "🏗️",
                "color": "#6b7280",
                "response_time_target": 30,
                "required_responders": "engineer,utility_crew"
            },
            {
                "name": "Other",
                "description": "Other types of incidents not covered above",
                "severity_weight": 1.0,
                "default_priority": 2,
                "icon": "📝",
                "color": "#9ca3af",
                "response_time_target": 30,
                "required_responders": "general_responder"
            }
        ]
        
        for cat_data in categories:
            existing = db.query(models.IncidentCategory).filter_by(name=cat_data["name"]).first()
            if not existing:
                category = models.IncidentCategory(**cat_data)
                db.add(category)
        
        # Add Calapan barangays
        barangays = [
            {
                "name": "Tawagan",
                "population": 4500,
                "area_sqkm": 2.5,
                "priority_level": 3,
                "has_hospital": False,
                "has_fire_station": True,
                "has_police_station": False,
                "barangay_captain": "Juan Dela Cruz",
                "contact_number": "09123456789",
                "emergency_contact": "288-1001",
                "latitude": 13.411,
                "longitude": 121.181
            },
            {
                "name": "Sta. Isabel",
                "population": 5200,
                "area_sqkm": 3.2,
                "priority_level": 4,
                "has_hospital": True,
                "has_fire_station": True,
                "has_police_station": True,
                "barangay_captain": "Maria Santos",
                "contact_number": "09123456790",
                "emergency_contact": "288-1002",
                "latitude": 13.415,
                "longitude": 121.185
            },
            {
                "name": "Lumangbayan",
                "population": 3800,
                "area_sqkm": 2.8,
                "priority_level": 2,
                "has_hospital": False,
                "has_fire_station": False,
                "has_police_station": False,
                "barangay_captain": "Pedro Reyes",
                "contact_number": "09123456791",
                "emergency_contact": "288-1003",
                "latitude": 13.408,
                "longitude": 121.178
            },
            {
                "name": "Poblacion",
                "population": 6800,
                "area_sqkm": 4.5,
                "priority_level": 5,
                "has_hospital": True,
                "has_fire_station": True,
                "has_police_station": True,
                "barangay_captain": "Ana Lopez",
                "contact_number": "09123456792",
                "emergency_contact": "288-1004",
                "latitude": 13.410,
                "longitude": 121.180
            },
            {
                "name": "Navotas",
                "population": 3200,
                "area_sqkm": 2.2,
                "priority_level": 2,
                "has_hospital": False,
                "has_fire_station": False,
                "has_police_station": False,
                "barangay_captain": "Carlos Garcia",
                "contact_number": "09123456793",
                "emergency_contact": "288-1005",
                "latitude": 13.413,
                "longitude": 121.183
            },
            {
                "name": "Santiago",
                "population": 4100,
                "area_sqkm": 3.0,
                "priority_level": 3,
                "has_hospital": False,
                "has_fire_station": True,
                "has_police_station": False,
                "barangay_captain": "Elena Torres",
                "contact_number": "09123456794",
                "emergency_contact": "288-1006",
                "latitude": 13.405,
                "longitude": 121.175
            }
        ]
        
        for brgy_data in barangays:
            existing = db.query(models.BarangayInfo).filter_by(name=brgy_data["name"]).first()
            if not existing:
                barangay = models.BarangayInfo(**brgy_data)
                db.add(barangay)
        
        # Add emergency facilities
        facilities = [
            {
                "name": "Oriental Mindoro Provincial Hospital",
                "facility_type": "hospital",
                "latitude": 13.405,
                "longitude": 121.175,
                "contact_number": "(043) 288-2000",
                "capacity": 200,
                "is_available": True,
                "open_24_7": True,
                "barangay": "Sta. Isabel",
                "address": "Sta. Isabel, Calapan City",
                "notes": "Main public hospital in Calapan"
            },
            {
                "name": "Calapan City Fire Station",
                "facility_type": "fire_station",
                "latitude": 13.410,
                "longitude": 121.182,
                "contact_number": "288-3333",
                "capacity": 50,
                "is_available": True,
                "open_24_7": True,
                "barangay": "Poblacion",
                "address": "Poblacion, Calapan City",
                "notes": "Main fire station"
            },
            {
                "name": "Calapan City Police Station",
                "facility_type": "police_station",
                "latitude": 13.412,
                "longitude": 121.180,
                "contact_number": "288-4444",
                "capacity": 100,
                "is_available": True,
                "open_24_7": True,
                "barangay": "Poblacion",
                "address": "Poblacion, Calapan City",
                "notes": "Main police station"
            },
            {
                "name": "Calapan City Rescue Unit",
                "facility_type": "rescue",
                "latitude": 13.411,
                "longitude": 121.181,
                "contact_number": "288-1111",
                "capacity": 30,
                "is_available": True,
                "open_24_7": True,
                "barangay": "Poblacion",
                "address": "City Hall Compound, Calapan City",
                "notes": "Emergency rescue services"
            }
        ]
        
        for fac_data in facilities:
            existing = db.query(models.EmergencyFacility).filter_by(name=fac_data["name"]).first()
            if not existing:
                facility = models.EmergencyFacility(**fac_data)
                db.add(facility)
        
        db.commit()
        print("✅ Default data added:")
        print("   - 7 incident categories")
        print("   - 6 barangays")
        print("   - 4 emergency facilities")
        
    except Exception as e:
        db.rollback()
        print(f"❌ Failed to add default data: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    migrate_incident_tables()