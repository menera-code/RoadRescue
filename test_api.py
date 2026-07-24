# test_openai_v2.py
import os
from dotenv import load_dotenv

load_dotenv()

print("Testing OpenAI v2.15.0...")

# Test 1: Check API key
api_key = os.getenv("OPENAI_API_KEY")
if api_key and api_key != "your-openai-api-key-here":
    print(f"✓ API Key found: {api_key[:10]}...")
else:
    print("✗ API Key not found or not set properly")

# Test 2: Test OpenAI import
try:
    from openai import OpenAI
    print("✓ OpenAI v2.x import successful")
    
    # Test 3: Initialize client
    client = OpenAI(api_key=api_key)
    print("✓ OpenAI client initialized")
    
    # Test 4: Simple test request (without actually calling API)
    print("✓ Ready to make API calls")
    
except Exception as e:
    print(f"✗ Error: {e}")

print("\nAll tests completed!")