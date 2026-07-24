from database import engine

try:
    conn = engine.connect()
    print("✅ MySQL connection successful")
    conn.close()
except Exception as e:
    print("❌ Database error:", e)
