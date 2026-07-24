# test_gemini_final.py
import os
from dotenv import load_dotenv
import google.genai as genai

load_dotenv()

def test_gemini_api():
    """Test Gemini API connection"""
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ GEMINI_API_KEY not found in .env file")
        return False
    
    try:
        # Initialize client
        client = genai.Client(api_key=api_key)
        
        # Test available models
        print("🔍 Testing available models...")
        models = client.models.list()
        
        # Find Gemini 2.5 models
        gemini_models = []
        for model in models:
            if "gemini-2.5" in model.name:
                gemini_models.append(model.name)
        
        print(f"✅ Found {len(gemini_models)} Gemini 2.5 models:")
        for model_name in gemini_models[:5]:
            print(f"  - {model_name}")
        
        if not gemini_models:
            print("❌ No Gemini 2.5 models found")
            return False
        
        # Use first available Gemini 2.5 model
        test_model = gemini_models[0]
        print(f"\n🧪 Testing with model: {test_model}")
        
        # Simple test
        response = client.models.generate_content(
            model=test_model,
            contents="Say 'Hello from Gemini in RESQAPP'",
            config=genai.types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=100
            )
        )
        
        print(f"✅ Gemini test successful!")
        print(f"Response: {response.text}")
        
        # Update .env file automatically
        update_env_file(test_model)
        return True
        
    except Exception as e:
        print(f"❌ Gemini test failed: {e}")
        return False

def update_env_file(model_name):
    """Update .env file with correct model name"""
    env_file = '.env'
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            lines = f.readlines()
        
        updated = False
        new_lines = []
        for line in lines:
            if line.startswith('GEMINI_MODEL='):
                new_lines.append(f'GEMINI_MODEL={model_name}\n')
                updated = True
            else:
                new_lines.append(line)
        
        if not updated:
            new_lines.append(f'\nGEMINI_MODEL={model_name}\n')
        
        with open(env_file, 'w') as f:
            f.writelines(new_lines)
        
        print(f"📝 Updated .env file with model: {model_name}")
    else:
        print(f"⚠️ Could not find .env file to update")

if __name__ == "__main__":
    test_gemini_api()