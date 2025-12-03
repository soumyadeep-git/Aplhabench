import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

def get_llm(temperature=0):
    """
    Factory to get the Gemini LLM instance.
    Uses flash for speed, but you can swap to 'gemini-1.5-pro' if
    reasoning requires it.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env")

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=temperature,
        google_api_key=api_key,
        convert_system_message_to_human=True
    )