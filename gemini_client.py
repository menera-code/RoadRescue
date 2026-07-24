import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
import google.genai as genai
import asyncio

load_dotenv()


class GeminiChatbot:
    def __init__(self):
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("⚠️ GEMINI_API_KEY not found, using fallback responses")
                self.client = None
                return

            self.client = genai.Client(api_key=api_key)
            self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            print(f"✅ Gemini chatbot initialized with model: {self.model}")

        except Exception as e:
            print(f"❌ Gemini initialization error: {e}")
            self.client = None

    def get_system_prompt(self) -> str:
        """
        Optimized system prompt – shorter, focuses on key instructions,
        reduces token usage while keeping essential legal references.
        """
        return (
            "You are RESQAPP Assistant for Calapan City's emergency system.\n"
            "Rules:\n"
            "1. For life-threatening emergencies → tell user to call 911 or local numbers: Rescue 288-1111, Fire 288-3333, Police 288-4444. Never delay.\n"
            "2. For incident reports → guide user to 'Create Report' feature.\n"
            "3. For traffic rules/fines → use RA 4136. Provide fine ranges (e.g., no helmet ₱1,500-₃,000). Advise checking latest LTO schedule.\n"
            "4. For legal or privacy concerns → mention RA 10173 (Data Privacy Act) and that info is confidential.\n"
            "5. Always include disclaimer: 'This is general guidance, not legal advice. Consult LTO/PNP/attorney for specifics.'\n"
            "Be concise, polite, and use bullet points when listing multiple items."
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_response(self, messages: List[Dict[str, str]]) -> str:
        """Get AI response from Gemini API."""
        if not self.client:
            return self.get_fallback_response(messages)

        if not messages:
            return "Hello! How can I assist you with emergencies, traffic rules, or reporting?"

        try:
            # Build content with system instruction as first user message (Gemini style)
            full_prompt = self.get_system_prompt() + "\n\nConversation:\n"
            for msg in messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                full_prompt += f"{role}: {msg['content']}\n"
            full_prompt += "Assistant:"

            response = self.client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=float(os.getenv("GEMINI_TEMPERATURE", 0.5)),  # lower = more focused
                    max_output_tokens=int(os.getenv("GEMINI_MAX_TOKENS", 800)),
                )
            )
            # Clean up response
            answer = response.text.strip()
            if not answer:
                return self.get_fallback_response(messages)
            return answer

        except Exception as e:
            print(f"⚠️ Gemini API error: {type(e).__name__}: {e}")
            return self.get_fallback_response(messages)

    def get_fallback_response(self, messages: List[Dict[str, str]]) -> str:
        """Fallback responses that are context‑aware and concise."""
        if not messages:
            return "How can I help you today? You can ask about reporting incidents, traffic rules, or emergency contacts."

        last_msg = messages[-1]["content"].lower()

        if any(word in last_msg for word in ["report", "incident", "accident"]):
            return "To report an incident, open 'Create Report' in the app. For immediate danger, call 911 or Calapan Rescue at 288-1111. Always provide truthful information – false reports are punishable by law."

        if any(word in last_msg for word in ["emergency", "help", "urgent"]):
            return "🚨 Emergency contacts:\n- Rescue: 288-1111\n- Fire: 288-3333\n- Police: 288-4444\n- 911 (nationwide)\nCall now if life‑threatening."

        if any(word in last_msg for word in ["traffic", "fine", "violation", "helmet", "license"]):
            return ("📜 Common traffic fines (RA 4136):\n"
                    "• No helmet (rider): ₱1,500 – ₱3,000\n"
                    "• No seat belt: ₱1,000\n"
                    "• Using phone while driving: ₱5,000\n"
                    "• No OR/CR: ₱10,000\n"
                    "• Reckless driving: ₱2,000 – ₱5,000\n"
                    "For exact amounts, contact LTO Calapan. This is not legal advice.")

        if "data privacy" in last_msg:
            return "Your personal data is protected under RA 10173 (Data Privacy Act). We only use it for incident response and never share without consent. You may request deletion via your profile."

        return ("I can help with emergencies, incident reporting, traffic rules (RA 4136), and Calapan City services. "
                "What would you like to know?")


# Singleton instance
chatbot = GeminiChatbot()